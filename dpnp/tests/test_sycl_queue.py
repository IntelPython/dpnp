import copy
import tempfile

import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

import dpnp
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    is_win_platform,
)

list_of_backend_str = [
    "host",
    "level_zero",
    "opencl",
]

list_of_device_type_str = [
    "host",
    "gpu",
    "cpu",
]

available_devices = [
    d for d in dpctl.get_devices() if not getattr(d, "has_aspect_host", False)
]

valid_devices = []
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
        valid_devices.append(device)


def assert_sycl_queue_equal(result, expected):
    assert result.backend == expected.backend
    assert result.sycl_context == expected.sycl_context
    assert result.sycl_device == expected.sycl_device
    assert result.is_in_order == expected.is_in_order
    assert result.has_enable_profiling == expected.has_enable_profiling
    exec_queue = dpctl.utils.get_execution_queue([result, expected])
    assert exec_queue is not None


def vvsort(val, vec, size, xp):
    val_kwargs = {}
    if hasattr(val, "sycl_queue"):
        val_kwargs["sycl_queue"] = getattr(val, "sycl_queue", None)

    vec_kwargs = {}
    if hasattr(vec, "sycl_queue"):
        vec_kwargs["sycl_queue"] = getattr(vec, "sycl_queue", None)

    for i in range(size):
        imax = i
        for j in range(i + 1, size):
            unravel_imax = numpy.unravel_index(imax, val.shape)
            unravel_j = numpy.unravel_index(j, val.shape)
            if xp.abs(val[unravel_imax]) < xp.abs(val[unravel_j]):
                imax = j

        unravel_i = numpy.unravel_index(i, val.shape)
        unravel_imax = numpy.unravel_index(imax, val.shape)

        # swap elements in val array
        temp = xp.array(val[unravel_i], dtype=vec.dtype, **val_kwargs)
        val[unravel_i] = val[unravel_imax]
        val[unravel_imax] = temp

        # swap corresponding columns in vec matrix
        temp = xp.array(vec[:, i], dtype=val.dtype, **vec_kwargs)
        vec[:, i] = vec[:, imax]
        vec[:, imax] = temp


@pytest.mark.parametrize(
    "func, arg, kwargs",
    [
        pytest.param("arange", [-25.7], {"stop": 10**8, "step": 15}),
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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_array_creation(func, arg, kwargs, device):
    dpnp_kwargs = dict(kwargs)
    dpnp_kwargs["device"] = device
    dpnp_array = getattr(dpnp, func)(*arg, **dpnp_kwargs)

    numpy_kwargs = dict(kwargs)
    numpy_kwargs["dtype"] = dpnp_array.dtype
    numpy_array = getattr(numpy, func)(*arg, **numpy_kwargs)

    assert_dtype_allclose(dpnp_array, numpy_array)
    assert dpnp_array.sycl_device == device


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_empty(device):
    dpnp_array = dpnp.empty((2, 2), device=device)
    assert dpnp_array.sycl_device == device


@pytest.mark.parametrize(
    "device_x",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "device_y",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_empty_like(device_x, device_y):
    x = dpnp.array([1, 2, 3], device=device_x)
    y = dpnp.empty_like(x)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)
    y = dpnp.empty_like(x, device=device_y)
    assert_sycl_queue_equal(y.sycl_queue, x.to_device(device_y).sycl_queue)


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        pytest.param("copy", ["x0"], {}),
        pytest.param("diag", ["x0"], {}),
        pytest.param("full_like", ["x0"], {"fill_value": 5}),
        pytest.param("geomspace", ["x0[0:3]", "8", "4"], {}),
        pytest.param("geomspace", ["1", "x0[2:4]", "4"], {}),
        pytest.param("linspace", ["x0[0:2]", "8", "4"], {}),
        pytest.param("linspace", ["0", "x0[2:4]", "4"], {}),
        pytest.param("logspace", ["x0[0:2]", "8", "4"], {}),
        pytest.param("logspace", ["0", "x0[2:4]", "4"], {}),
        pytest.param("ones_like", ["x0"], {}),
        pytest.param("tril", ["x0.reshape((2,2))"], {}),
        pytest.param("triu", ["x0.reshape((2,2))"], {}),
        pytest.param("linspace", ["x0", "4", "4"], {}),
        pytest.param("linspace", ["1", "x0", "4"], {}),
        pytest.param("vander", ["x0"], {}),
        pytest.param("zeros_like", ["x0"], {}),
    ],
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_array_creation_follow_device(func, args, kwargs, device):
    x_orig = numpy.array([1, 2, 3, 4])
    numpy_args = [eval(val, {"x0": x_orig}) for val in args]
    y_orig = getattr(numpy, func)(*numpy_args, **kwargs)

    x = dpnp.array([1, 2, 3, 4], device=device)
    dpnp_args = [eval(val, {"x0": x}) for val in args]

    y = getattr(dpnp, func)(*dpnp_args, **kwargs)
    assert_allclose(y_orig, y, rtol=1e-04)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)


@pytest.mark.skipif(
    numpy.lib.NumpyVersion(numpy.__version__) < "1.25.0",
    reason="numpy.logspace supports a non-scalar base argument since 1.25.0",
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_array_creation_follow_device_logspace_base(device):
    x_orig = numpy.array([1, 2, 3, 4])
    y_orig = numpy.logspace(0, 8, 4, base=x_orig[1:3])

    x = dpnp.array([1, 2, 3, 4], device=device)
    y = dpnp.logspace(0, 8, 4, base=x[1:3])

    assert_allclose(y_orig, y, rtol=1e-04)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        pytest.param("diag", ["x0"], {}),
        pytest.param("diagflat", ["x0"], {}),
    ],
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_array_creation_follow_device_2d_array(func, args, kwargs, device):
    x_orig = numpy.arange(9).reshape(3, 3)
    numpy_args = [eval(val, {"x0": x_orig}) for val in args]
    y_orig = getattr(numpy, func)(*numpy_args, **kwargs)

    x = dpnp.arange(9, device=device).reshape(3, 3)
    dpnp_args = [eval(val, {"x0": x}) for val in args]

    y = getattr(dpnp, func)(*dpnp_args, **kwargs)
    assert_allclose(y_orig, y)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        pytest.param("copy", ["x0"], {}),
        pytest.param("diag", ["x0"], {}),
        pytest.param("full", ["10", "x0[3]"], {}),
        pytest.param("full_like", ["x0"], {"fill_value": 5}),
        pytest.param("ones_like", ["x0"], {}),
        pytest.param("zeros_like", ["x0"], {}),
        pytest.param("linspace", ["x0", "4", "4"], {}),
        pytest.param("linspace", ["1", "x0", "4"], {}),
        pytest.param("vander", ["x0"], {}),
    ],
)
@pytest.mark.parametrize(
    "device_x",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "device_y",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_array_creation_cross_device(func, args, kwargs, device_x, device_y):
    if func == "linspace" and is_win_platform():
        pytest.skip("CPU driver experiences an instability on Windows.")

    x_orig = numpy.array([1, 2, 3, 4])
    numpy_args = [eval(val, {"x0": x_orig}) for val in args]
    y_orig = getattr(numpy, func)(*numpy_args, **kwargs)

    x = dpnp.array([1, 2, 3, 4], device=device_x)
    dpnp_args = [eval(val, {"x0": x}) for val in args]

    dpnp_kwargs = dict(kwargs)
    y = getattr(dpnp, func)(*dpnp_args, **dpnp_kwargs)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)

    dpnp_kwargs["device"] = device_y
    y = getattr(dpnp, func)(*dpnp_args, **dpnp_kwargs)
    assert_allclose(y_orig, y)

    assert_sycl_queue_equal(y.sycl_queue, x.to_device(device_y).sycl_queue)


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        pytest.param("diag", ["x0"], {}),
        pytest.param("diagflat", ["x0"], {}),
    ],
)
@pytest.mark.parametrize(
    "device_x",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "device_y",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_array_creation_cross_device_2d_array(
    func, args, kwargs, device_x, device_y
):
    if func == "linspace" and is_win_platform():
        pytest.skip("CPU driver experiences an instability on Windows.")

    x_orig = numpy.arange(9).reshape(3, 3)
    numpy_args = [eval(val, {"x0": x_orig}) for val in args]
    y_orig = getattr(numpy, func)(*numpy_args, **kwargs)

    x = dpnp.arange(9, device=device_x).reshape(3, 3)
    dpnp_args = [eval(val, {"x0": x}) for val in args]

    dpnp_kwargs = dict(kwargs)
    y = getattr(dpnp, func)(*dpnp_args, **dpnp_kwargs)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)

    dpnp_kwargs["device"] = device_y
    y = getattr(dpnp, func)(*dpnp_args, **dpnp_kwargs)
    assert_allclose(y_orig, y)

    assert_sycl_queue_equal(y.sycl_queue, x.to_device(device_y).sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_array_creation_from_file(device):
    with tempfile.TemporaryFile() as fh:
        fh.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08")
        fh.flush()

        fh.seek(0)
        numpy_array = numpy.fromfile(fh)

        fh.seek(0)
        dpnp_array = dpnp.fromfile(fh, device=device)

    assert_dtype_allclose(dpnp_array, numpy_array)
    assert dpnp_array.sycl_device == device


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_array_creation_load_txt(device):
    with tempfile.TemporaryFile() as fh:
        fh.write(b"1 2 3 4")
        fh.flush()

        fh.seek(0)
        numpy_array = numpy.loadtxt(fh)

        fh.seek(0)
        dpnp_array = dpnp.loadtxt(fh, device=device)

    assert_dtype_allclose(dpnp_array, numpy_array)
    assert dpnp_array.sycl_device == device


@pytest.mark.parametrize(
    "device_x",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "device_y",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_copy_method(device_x, device_y):
    x = dpnp.array([[1, 2, 3], [4, 5, 6]], device=device_x)

    y = x.copy()
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)

    q = dpctl.SyclQueue(device_y)
    y = x.copy(sycl_queue=q)
    assert_sycl_queue_equal(y.sycl_queue, q)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_copy_operation(device):
    x = dpnp.array([[1, 2, 3], [4, 5, 6]], device=device)

    y = copy.copy(x)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_extract(device):
    x = dpnp.arange(3, device=device)
    y = dpnp.array([True, False, True], device=device)
    result = dpnp.extract(x, y)

    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, y.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_meshgrid(device):
    x = dpnp.arange(100, device=device)
    y = dpnp.arange(100, device=device)
    z = dpnp.meshgrid(x, y)
    assert_sycl_queue_equal(z[0].sycl_queue, x.sycl_queue)
    assert_sycl_queue_equal(z[1].sycl_queue, y.sycl_queue)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
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
        pytest.param("count_nonzero", [3, 0, 2, -1.2]),
        pytest.param("cumprod", [[1, 2, 3], [4, 5, 6]]),
        pytest.param("cumsum", [[1, 2, 3], [4, 5, 6]]),
        pytest.param("cumulative_prod", [1, 2, 3, 4, 5, 6]),
        pytest.param("cumulative_sum", [1, 2, 3, 4, 5, 6]),
        pytest.param("degrees", [numpy.pi, numpy.pi / 2, 0]),
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
        pytest.param("ptp", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("radians", [180, 90, 45, 0]),
        pytest.param(
            "real", [complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)]
        ),
        pytest.param("real_if_close", [2.1 + 4e-15j, 5.2 + 3e-16j]),
        pytest.param("reciprocal", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("rot90", [[1, 2], [3, 4]]),
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
        pytest.param(
            "trace", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        pytest.param("trapezoid", [1, 2, 3]),
        pytest.param("trim_zeros", [0, 0, 0, 1, 2, 3, 0, 2, 1, 0]),
        pytest.param("trunc", [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
        pytest.param("unwrap", [[0, 1, 2, -1, 0]]),
        pytest.param("var", [1.0, 2.0, 4.0, 7.0]),
    ],
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_1in_1out(func, data, device):
    x = dpnp.array(data, device=device)
    result = getattr(dpnp, func)(x)

    if (
        func == "trapezoid"
        and numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0"
    ):
        # `trapezoid` is available from NumPy 2.0
        func = "trapz"

    if (
        func in ["cumulative_prod", "cumulative_sum"]
        and numpy.lib.NumpyVersion(numpy.__version__) < "2.1.0"
    ):
        pytest.skip(
            "cumulative_prod and cumulative_sum are available from NumPy 2.1"
        )

    x_orig = dpnp.asnumpy(x)
    expected = getattr(numpy, func)(x_orig)
    assert_dtype_allclose(result, expected)

    expected_queue = x.sycl_queue
    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_logic_op_1in(op, device):
    x = dpnp.array(
        [-dpnp.inf, -1.0, 0.0, 1.0, dpnp.inf, dpnp.nan], device=device
    )
    result = getattr(dpnp, op)(x)

    x_orig = dpnp.asnumpy(x)
    expected = getattr(numpy, op)(x_orig)
    assert_dtype_allclose(result, expected)

    expected_queue = x.sycl_queue
    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_proj(device):
    X = [
        complex(1, 2),
        complex(dpnp.inf, -1),
        complex(0, -dpnp.inf),
        complex(-dpnp.inf, dpnp.nan),
    ]
    Y = [
        complex(1, 2),
        complex(dpnp.inf, -0.0),
        complex(dpnp.inf, -0.0),
        complex(dpnp.inf, 0.0),
    ]

    x = dpnp.array(X, device=device)
    result = dpnp.proj(x)
    expected = dpnp.array(Y)
    assert_allclose(result, expected)

    expected_queue = x.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_rsqrt(device):
    X = [1.0, 8.0, 27.0]
    x = dpnp.array(X, device=device)
    result = dpnp.rsqrt(x)
    expected = 1 / numpy.sqrt(x.asnumpy())
    assert_allclose(result, expected)

    expected_queue = x.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_logsumexp(device):
    x = dpnp.arange(10, device=device)
    result = dpnp.logsumexp(x)
    expected = numpy.logaddexp.reduce(x.asnumpy())
    assert_dtype_allclose(result, expected)

    expected_queue = x.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_cumlogsumexp(device):
    x = dpnp.arange(10, device=device)
    result = dpnp.cumlogsumexp(x)
    expected = numpy.logaddexp.accumulate(x.asnumpy())
    assert_dtype_allclose(result, expected)

    expected_queue = x.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_reduce_hypot(device):
    x = dpnp.arange(10, device=device)
    result = dpnp.reduce_hypot(x)
    expected = numpy.hypot.reduce(x.asnumpy())
    assert_dtype_allclose(result, expected)

    expected_queue = x.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


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
        pytest.param(
            "corrcoef",
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ),
        pytest.param("correlate", [1, 2, 3], [4, 5, 6]),
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
            "float_power",
            [0, 1, 2, 3, 4, 5],
            [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
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
        pytest.param(
            "gcd",
            [0, 1, 2, 3, 4, 5],
            [20, 20, 20, 20, 20, 20],
        ),
        pytest.param(
            "gradient",
            [1.0, 2.0, 4.0, 7.0, 11.0, 16.0],
            [0.0, 1.0, 1.5, 3.5, 4.0, 6.0],
        ),
        pytest.param("heaviside", [-1.5, 0, 2.0], [0.5]),
        pytest.param(
            "histogram_bin_edges",
            [0, 0, 0, 1, 2, 3, 3, 4, 5],
            [1, 2],
        ),
        pytest.param(
            "hypot", [[1.0, 2.0, 3.0, 4.0]], [[-1.0, -2.0, -4.0, -5.0]]
        ),
        pytest.param("inner", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        pytest.param("kron", [3.0, 4.0, 5.0], [1.0, 2.0]),
        pytest.param(
            "lcm",
            [0, 1, 2, 3, 4, 5],
            [20, 20, 20, 20, 20, 20],
        ),
        pytest.param(
            "ldexp",
            [5, 5, 5, 5, 5],
            [0, 1, 2, 3, 4],
        ),
        pytest.param("logaddexp", [[-1, 2, 5, 9]], [[4, -3, 2, -8]]),
        pytest.param("logaddexp2", [[-1, 2, 5, 9]], [[4, -3, 2, -8]]),
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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_2in_1out(func, data1, data2, device):
    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)(x1, x2)

    if (
        func == "trapezoid"
        and numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0"
    ):
        # `trapezoid` is available from NumPy 2.0
        func = "trapz"

    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = getattr(numpy, func)(x1_orig, x2_orig)

    assert_dtype_allclose(result, expected)

    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, x2.sycl_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_logic_op_2in(op, device):
    x1 = dpnp.array(
        [-dpnp.inf, -1.0, 0.0, 1.0, dpnp.inf, dpnp.nan], device=device
    )
    x2 = dpnp.array(
        [dpnp.inf, 1.0, 0.0, -1.0, -dpnp.inf, dpnp.nan], device=device
    )
    # Remove NaN value from input arrays because numpy raises RuntimeWarning
    if op in [
        "greater",
        "greater_equal",
        "less",
        "less_equal",
    ]:
        x1 = x1[:-1]
        x2 = x2[:-1]
    result = getattr(dpnp, op)(x1, x2)

    x1_orig = dpnp.asnumpy(x1)
    x2_orig = dpnp.asnumpy(x2)
    expected = getattr(numpy, op)(x1_orig, x2_orig)

    assert_dtype_allclose(result, expected)

    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, x2.sycl_queue)


@pytest.mark.parametrize(
    "func, data, scalar",
    [
        pytest.param("searchsorted", [11, 12, 13, 14, 15], 13),
    ],
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_2in_with_scalar_1out(func, data, scalar, device):
    x1_orig = numpy.array(data)
    expected = getattr(numpy, func)(x1_orig, scalar)

    x1 = dpnp.array(data, device=device)
    result = getattr(dpnp, func)(x1, scalar)

    assert_allclose(result, expected)
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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_broadcasting(func, data1, data2, device):
    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = getattr(numpy, func)(x1_orig, x2_orig)

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)(x1, x2)

    assert_array_equal(result, expected)

    expected_queue = x1.sycl_queue
    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_2in_1out_diff_queue_but_equal_context(func, device):
    x1 = dpnp.arange(10)
    x2 = dpnp.arange(10, sycl_queue=dpctl.SyclQueue(device))[::-1]
    with assert_raises((ValueError, ExecutionPlacementError)):
        getattr(dpnp, func)(x1, x2)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "shape_pair",
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
def test_matmul(device, shape_pair):
    shape1, shape2 = shape_pair
    a1 = numpy.arange(numpy.prod(shape1)).reshape(shape1)
    a2 = numpy.arange(numpy.prod(shape2)).reshape(shape2)

    b1 = dpnp.asarray(a1, device=device)
    b2 = dpnp.asarray(a2, device=device)

    result = dpnp.matmul(b1, b2)
    expected = numpy.matmul(a1, a2)
    assert_allclose(expected, result)

    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, b1.sycl_queue)
    assert_sycl_queue_equal(result_queue, b2.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "shape_pair",
    [
        ((4,), (4,)),  # call_flag: dot
        ((3, 1), (3, 1)),
        ((2, 0), (2, 0)),  # zero-size inputs, 1D output
        ((3, 0, 4), (3, 0, 4)),  # zero-size output
        ((3, 4), (3, 4)),  # call_flag: vecdot
    ],
)
def test_vecdot(device, shape_pair):
    shape1, shape2 = shape_pair
    a1 = numpy.arange(numpy.prod(shape1)).reshape(shape1)
    a2 = numpy.arange(numpy.prod(shape2)).reshape(shape2)

    b1 = dpnp.asarray(a1, device=device)
    b2 = dpnp.asarray(a2, device=device)

    result = dpnp.vecdot(b1, b2)
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, b1.sycl_queue)
    assert_sycl_queue_equal(result_queue, b2.sycl_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
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


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize(
    "func,data",
    [
        pytest.param("sqrt", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    ],
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_out_1in_1out(func, data, device):
    x_orig = numpy.array(data)
    np_out = getattr(numpy, func)(x_orig)
    expected = numpy.empty_like(np_out)
    getattr(numpy, func)(x_orig, out=expected)

    x = dpnp.array(data, device=device)
    dp_out = getattr(dpnp, func)(x)
    result = dpnp.empty_like(dp_out)
    getattr(dpnp, func)(x, out=result)

    assert_allclose(result, expected)

    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize(
    "func,data1,data2",
    [
        pytest.param(
            "add",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        ),
        pytest.param("copysign", [0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]),
        pytest.param(
            "divide", [0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 4.0, 4.0, 4.0, 4.0]
        ),
        # dpnp.dot has 3 different implementations based on input arrays dtype
        # checking all of them
        pytest.param("dot", [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]),
        pytest.param("dot", [3, 4, 5], [1, 2, 3]),
        pytest.param("dot", [3 + 2j, 4 + 1j, 5], [1, 2 + 3j, 3]),
        pytest.param(
            "floor_divide", [1.0, 2.0, 3.0, 4.0], [2.5, 2.5, 2.5, 2.5]
        ),
        pytest.param(
            "fmod",
            [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        pytest.param("maximum", [2.0, 3.0, 4.0], [1.0, 5.0, 2.0]),
        pytest.param("minimum", [2.0, 3.0, 4.0], [1.0, 5.0, 2.0]),
        pytest.param(
            "multiply",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
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
        pytest.param(
            "subtract",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        ),
    ],
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_out_2in_1out(func, data1, data2, device):
    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    np_out = getattr(numpy, func)(x1_orig, x2_orig)
    expected = numpy.empty_like(np_out)
    getattr(numpy, func)(x1_orig, x2_orig, out=expected)

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    dp_out = getattr(dpnp, func)(x1, x2)
    result = dpnp.empty_like(dp_out)
    getattr(dpnp, func)(x1, x2, out=result)

    assert_allclose(result, expected)

    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, x2.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_modf(device):
    data = [0, 3.5]

    x_orig = numpy.array(data)
    expected1, expected2 = numpy.modf(x_orig)

    x = dpnp.array(data, device=device)
    result1, result2 = dpnp.modf(x)

    assert_array_equal(result1, expected1)
    assert_array_equal(result2, expected2)

    expected_queue = x.sycl_queue
    result1_queue = result1.sycl_queue
    result2_queue = result2.sycl_queue

    assert_sycl_queue_equal(result1_queue, expected_queue)
    assert_sycl_queue_equal(result2_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_multi_dot(device):
    numpy_array_list = []
    dpnp_array_list = []
    for num_array in [3, 5]:  # number of arrays in multi_dot
        for _ in range(num_array):  # creat arrays one by one
            a = numpy.random.rand(10, 10)
            b = dpnp.array(a, device=device)

            numpy_array_list.append(a)
            dpnp_array_list.append(b)

        result = dpnp.linalg.multi_dot(dpnp_array_list)
        expected = numpy.linalg.multi_dot(numpy_array_list)
        assert_dtype_allclose(result, expected)

        _, exec_q = get_usm_allocations(dpnp_array_list)
        assert_sycl_queue_equal(result.sycl_queue, exec_q)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_einsum(device):
    numpy_array_list = []
    dpnp_array_list = []
    for _ in range(3):  # creat arrays one by one
        a = numpy.random.rand(10, 10)
        b = dpnp.array(a, device=device)

        numpy_array_list.append(a)
        dpnp_array_list.append(b)

    result = dpnp.einsum("ij,jk,kl->il", *dpnp_array_list)
    expected = numpy.einsum("ij,jk,kl->il", *numpy_array_list)
    assert_dtype_allclose(result, expected)

    _, exec_q = get_usm_allocations(dpnp_array_list)
    assert_sycl_queue_equal(result.sycl_queue, exec_q)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_out_multi_dot(device):
    numpy_array_list = []
    dpnp_array_list = []
    for num_array in [3, 5]:  # number of arrays in multi_dot
        for _ in range(num_array):  # creat arrays one by one
            a = numpy.random.rand(10, 10)
            b = dpnp.array(a, device=device)

            numpy_array_list.append(a)
            dpnp_array_list.append(b)

        dp_out = dpnp.empty((10, 10), device=device)
        result = dpnp.linalg.multi_dot(dpnp_array_list, out=dp_out)
        assert result is dp_out
        expected = numpy.linalg.multi_dot(numpy_array_list)
        assert_dtype_allclose(result, expected)

        _, exec_q = get_usm_allocations(dpnp_array_list)
        assert_sycl_queue_equal(result.sycl_queue, exec_q)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
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
    dpnp_data = dpnp.arange(100, device=device)
    expected_queue = dpnp_data.sycl_queue
    for mode in all_modes:
        result = dpnp.pad(dpnp_data, (25, 20), mode=mode)
        result_queue = result.sycl_queue
        assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_require(device):
    dpnp_data = dpnp.arange(10, device=device).reshape(2, 5)
    result = dpnp.require(dpnp_data, dtype="f4", requirements=["F"])

    expected_queue = dpnp_data.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)

    # No requirements
    result = dpnp.require(dpnp_data, dtype="f4")
    expected_queue = dpnp_data.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_resize(device):
    dpnp_data = dpnp.arange(10, device=device)
    result = dpnp.resize(dpnp_data, (2, 5))

    expected_queue = dpnp_data.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


class TestFft:
    @pytest.mark.parametrize(
        "func", ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"]
    )
    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    def test_fft(self, func, device):
        dtype = numpy.float64 if func in ["rfft", "ihfft"] else numpy.complex128
        data = numpy.arange(20, dtype=dtype)
        dpnp_data = dpnp.array(data, device=device)

        expected = getattr(numpy.fft, func)(data)
        result = getattr(dpnp.fft, func)(dpnp_data)
        assert_dtype_allclose(result, expected, factor=24)

        expected_queue = dpnp_data.sycl_queue
        result_queue = result.sycl_queue
        assert_sycl_queue_equal(result_queue, expected_queue)

    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    def test_fftn(self, device):
        data = numpy.arange(24, dtype=numpy.complex64).reshape(2, 3, 4)
        dpnp_data = dpnp.array(data, device=device)

        expected = numpy.fft.fftn(data)
        result = dpnp.fft.fftn(dpnp_data)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        expected_queue = dpnp_data.sycl_queue
        result_queue = result.sycl_queue
        assert_sycl_queue_equal(result_queue, expected_queue)

        expected = numpy.fft.ifftn(expected)
        result = dpnp.fft.ifftn(result)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        result_queue = result.sycl_queue
        assert_sycl_queue_equal(result_queue, expected_queue)

    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    def test_rfftn(self, device):
        data = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
        dpnp_data = dpnp.array(data, device=device)

        expected = numpy.fft.rfftn(data)
        result = dpnp.fft.rfftn(dpnp_data)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        expected_queue = dpnp_data.sycl_queue
        result_queue = result.sycl_queue
        assert_sycl_queue_equal(result_queue, expected_queue)

        expected = numpy.fft.irfftn(expected)
        result = dpnp.fft.irfftn(result)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        result_queue = result.sycl_queue
        assert_sycl_queue_equal(result_queue, expected_queue)

    @pytest.mark.parametrize("func", ["fftfreq", "rfftfreq"])
    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    def test_fftfreq(self, func, device):
        result = getattr(dpnp.fft, func)(10, 0.5, device=device)
        expected = getattr(numpy.fft, func)(10, 0.5)

        assert_dtype_allclose(result, expected)
        assert result.sycl_device == device

    @pytest.mark.parametrize("func", ["fftshift", "ifftshift"])
    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    def test_fftshift(self, func, device):
        dpnp_data = dpnp.fft.fftfreq(10, 0.5, device=device)
        data = dpnp_data.asnumpy()

        expected = getattr(numpy.fft, func)(data)
        result = getattr(dpnp.fft, func)(dpnp_data)

        assert_dtype_allclose(result, expected)

        expected_queue = dpnp_data.sycl_queue
        result_queue = result.sycl_queue

        assert_sycl_queue_equal(result_queue, expected_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_cholesky(data, is_empty, device):
    if is_empty:
        numpy_data = numpy.empty(data, dtype=dpnp.default_float_type(device))
    else:
        numpy_data = numpy.array(data, dtype=dpnp.default_float_type(device))
    dpnp_data = dpnp.array(numpy_data, device=device)

    result = dpnp.linalg.cholesky(dpnp_data)
    expected = numpy.linalg.cholesky(numpy_data)
    assert_dtype_allclose(result, expected)

    expected_queue = dpnp_data.sycl_queue
    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize("p", [None, -dpnp.inf, -2, -1, 1, 2, dpnp.inf, "fro"])
def test_cond(device, p):
    numpy.random.seed(42)
    a = numpy.array(numpy.random.uniform(-5, 5, 16)).reshape(4, 4)
    ia = dpnp.array(a, device=device)

    result = dpnp.linalg.cond(ia, p=p)
    expected = numpy.linalg.cond(a, p=p)
    assert_dtype_allclose(result, expected)

    expected_queue = ia.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_det(device):
    data = [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.det(dpnp_data)
    expected = numpy.linalg.det(numpy_data)
    assert_allclose(expected, result)

    expected_queue = dpnp_data.sycl_queue
    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "func",
    [
        "eig",
        "eigvals",
        "eigh",
        "eigvalsh",
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (0, 0),
        (2, 3, 3),
        (0, 2, 2),
        (1, 0, 0),
    ],
    ids=[
        "(4, 4)",
        "(0, 0)",
        "(2, 3, 3)",
        "(0, 2, 2)",
        "(1, 0, 0)",
    ],
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_eigenvalue(func, shape, device):
    dtype = dpnp.default_float_type(device)
    # Set a `hermitian` flag for generate_random_numpy_array() to
    # get a symmetric array for eigh() and eigvalsh() or
    # non-symmetric for eig() and eigvals()
    is_hermitian = func in ("eigh, eigvalsh")
    # Set seed_value=81 to prevent
    # random generation of the input singular matrix
    a = generate_random_numpy_array(
        shape, dtype, hermitian=is_hermitian, seed_value=81, low=-5, high=5
    )
    dp_a = dpnp.array(a, device=device)

    expected_queue = dp_a.sycl_queue

    if func in ("eig", "eigh"):
        dp_val, dp_vec = getattr(dpnp.linalg, func)(dp_a)
        np_val, np_vec = getattr(numpy.linalg, func)(a)

        # Check the eigenvalue decomposition
        if a.ndim == 2:
            assert_allclose(
                dp_a @ dp_vec, dp_vec @ dpnp.diag(dp_val), rtol=1e-5, atol=1e-5
            )
        else:  # a.ndim == 3
            for i in range(a.shape[0]):
                assert_allclose(
                    dp_a[i].dot(dp_vec[i]),
                    dp_val[i] * dp_vec[i],
                    rtol=1e-5,
                    atol=1e-5,
                )
        assert dp_vec.shape == np_vec.shape
        assert dp_vec.dtype == np_vec.dtype

        dpnp_vec_queue = dp_vec.sycl_queue
        # compare queue and device
        assert_sycl_queue_equal(dpnp_vec_queue, expected_queue)

    else:  # eighvals or eigvalsh
        dp_val = getattr(dpnp.linalg, func)(dp_a)
        np_val = getattr(numpy.linalg, func)(a)

    assert_allclose(dp_val, np_val, rtol=1e-05, atol=1e-05)
    assert dp_val.shape == np_val.shape
    assert dp_val.dtype == np_val.dtype

    dpnp_val_queue = dp_val.sycl_queue
    # compare queue and device
    assert_sycl_queue_equal(dpnp_val_queue, expected_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_inv(shape, is_empty, device):
    if is_empty:
        numpy_x = numpy.empty(shape, dtype=dpnp.default_float_type(device))
    else:
        count_elem = numpy.prod(shape)
        numpy_x = numpy.arange(
            1, count_elem + 1, dtype=dpnp.default_float_type()
        ).reshape(shape)

    dpnp_x = dpnp.array(numpy_x, device=device)

    result = dpnp.linalg.inv(dpnp_x)
    expected = numpy.linalg.inv(numpy_x)
    assert_dtype_allclose(result, expected)

    expected_queue = dpnp_x.sycl_queue
    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize("n", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_matrix_power(n, device):
    data = numpy.array([[1, 2], [3, 5]], dtype=dpnp.default_float_type(device))
    dp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.matrix_power(dp_data, n)
    expected = numpy.linalg.matrix_power(data, n)
    assert_dtype_allclose(result, expected)

    expected_queue = dp_data.sycl_queue
    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_matrix_rank(data, tol, device):
    dp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.matrix_rank(dp_data, tol=tol)
    expected = numpy.linalg.matrix_rank(data, tol=tol)
    assert_array_equal(expected, result)

    expected_queue = dp_data.sycl_queue
    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.usefixtures("suppress_divide_numpy_warnings")
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "ord", [None, -dpnp.inf, -2, -1, 1, 2, 3, dpnp.inf, "fro", "nuc"]
)
@pytest.mark.parametrize(
    "axis",
    [-1, 0, 1, (0, 1), (-2, -1), None],
    ids=["-1", "0", "1", "(0, 1)", "(-2, -1)", "None"],
)
def test_norm(device, ord, axis):
    a = numpy.arange(120).reshape(2, 3, 4, 5)
    ia = dpnp.array(a, device=device)
    if (axis in [-1, 0, 1] and ord in ["nuc", "fro"]) or (
        isinstance(axis, tuple) and ord == 3
    ):
        pytest.skip("Invalid norm order for vectors.")
    elif axis is None and ord is not None:
        pytest.skip("Improper number of dimensions to norm")
    else:
        result = dpnp.linalg.norm(ia, ord=ord, axis=axis)
        expected = numpy.linalg.norm(a, ord=ord, axis=axis)
        assert_dtype_allclose(result, expected)

        expected_queue = ia.sycl_queue
        result_queue = result.sycl_queue
        assert_sycl_queue_equal(result_queue, expected_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_qr(shape, mode, device):
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

        dp_q_queue = dp_q.sycl_queue
        dp_r_queue = dp_r.sycl_queue

        assert_sycl_queue_equal(dp_q_queue, expected_queue)
        assert_sycl_queue_equal(dp_r_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize("full_matrices", [True, False], ids=["True", "False"])
@pytest.mark.parametrize("compute_uv", [True, False], ids=["True", "False"])
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
def test_svd(shape, full_matrices, compute_uv, device):
    dtype = dpnp.default_float_type(device)

    count_elems = numpy.prod(shape)
    dpnp_data = dpnp.arange(count_elems, dtype=dtype, device=device).reshape(
        shape
    )
    expected_queue = dpnp_data.sycl_queue

    if compute_uv:
        dpnp_u, dpnp_s, dpnp_vt = dpnp.linalg.svd(
            dpnp_data, full_matrices=full_matrices, compute_uv=compute_uv
        )

        dpnp_u_queue = dpnp_u.sycl_queue
        dpnp_vt_queue = dpnp_vt.sycl_queue
        dpnp_s_queue = dpnp_s.sycl_queue

        assert_sycl_queue_equal(dpnp_u_queue, expected_queue)
        assert_sycl_queue_equal(dpnp_vt_queue, expected_queue)
        assert_sycl_queue_equal(dpnp_s_queue, expected_queue)

    else:
        dpnp_s = dpnp.linalg.svd(
            dpnp_data, full_matrices=full_matrices, compute_uv=compute_uv
        )
        dpnp_s_queue = dpnp_s.sycl_queue

        assert_sycl_queue_equal(dpnp_s_queue, expected_queue)


class TestToDevice:
    @pytest.mark.parametrize(
        "device_from",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    @pytest.mark.parametrize(
        "device_to",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
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


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
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
    if numpy.lib.NumpyVersion(numpy.__version__) >= "2.0.0":
        pytest.skip("numpy.asfarray was removed")

    data = numpy.ones(100)
    dpnp_data = getattr(dpnp, func)(data, device=device)

    kwargs_items = {"device": device_param, "sycl_queue": queue_param}.items()
    kwargs = {
        k: getattr(dpnp_data, v, None) for k, v in kwargs_items if v != ""
    }

    result = dpnp.array(dpnp_data, **kwargs)

    assert_sycl_queue_equal(result.sycl_queue, dpnp_data.sycl_queue)


@pytest.mark.parametrize("copy", [True, False, None])
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_array_creation_from_dpctl(copy, device):
    dpt_data = dpt.ones((3, 3), device=device)

    result = dpnp.array(dpt_data, copy=copy)

    assert_sycl_queue_equal(result.sycl_queue, dpt_data.sycl_queue)
    assert isinstance(result, dpnp_array)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
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


@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
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
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_broadcast_to(device):
    x = dpnp.arange(5, device=device)
    y = dpnp.broadcast_to(x, (3, 5))
    assert_sycl_queue_equal(x.sycl_queue, y.sycl_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_concat_stack(func, data1, data2, device):
    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = getattr(numpy, func)((x1_orig, x2_orig))

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)((x1, x2))

    assert_allclose(result, expected)
    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, x2.sycl_queue)


class TestDelete:
    @pytest.mark.parametrize(
        "obj",
        [slice(None, None, 2), 3, [2, 3]],
        ids=["slice", "scalar", "list"],
    )
    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    def test_delete(self, obj, device):
        x = dpnp.arange(5, device=device)
        result = dpnp.delete(x, obj)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    def test_obj_ndarray(self, device):
        x = dpnp.arange(5, device=device)
        y = dpnp.array([1, 4], device=device)
        result = dpnp.delete(x, y)

        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)
        assert_sycl_queue_equal(result.sycl_queue, y.sycl_queue)


class TestInsert:
    @pytest.mark.parametrize(
        "obj",
        [slice(None, None, 2), 3, [2, 3]],
        ids=["slice", "scalar", "list"],
    )
    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
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
    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    def test_values_ndarray(self, obj, device):
        x = dpnp.arange(5, device=device)
        y = dpnp.array([1, 4], device=device)
        result = dpnp.insert(x, obj, y)

        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)
        assert_sycl_queue_equal(result.sycl_queue, y.sycl_queue)

    @pytest.mark.parametrize("values", [-2, [-1, -2]], ids=["scalar", "list"])
    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
    def test_obj_ndarray(self, values, device):
        x = dpnp.arange(5, device=device)
        y = dpnp.array([1, 4], device=device)
        result = dpnp.insert(x, y, values)

        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)
        assert_sycl_queue_equal(result.sycl_queue, y.sycl_queue)

    @pytest.mark.parametrize(
        "device",
        valid_devices,
        ids=[device.filter_string for device in valid_devices],
    )
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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_split(func, data1, device):
    x1_orig = numpy.array(data1)
    x1 = dpnp.array(data1, device=device)
    expected = getattr(numpy, func)(x1_orig, 2)
    result = getattr(dpnp, func)(x1, 2)

    assert_allclose(result[0], expected[0])
    assert_allclose(result[1], expected[1])
    assert_sycl_queue_equal(result[0].sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result[1].sycl_queue, x1.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_apply_along_axis(device):
    x = dpnp.arange(9, device=device).reshape(3, 3)
    result = dpnp.apply_along_axis(dpnp.sum, 0, x)

    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_apply_over_axes(device):
    x = dpnp.arange(18, device=device).reshape(2, 3, 3)
    result = dpnp.apply_over_axes(dpnp.sum, x, [0, 1])

    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize(
    "device_x",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "device_y",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_asarray(device_x, device_y):
    x = dpnp.array([1, 2, 3], device=device_x)
    y = dpnp.asarray([x, x, x], device=device_y)
    assert_sycl_queue_equal(y.sycl_queue, x.to_device(device_y).sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"prepend": 7}),
        pytest.param({"append": -2}),
        pytest.param({"prepend": -4, "append": 5}),
    ],
)
def test_diff_scalar_append(device, kwargs):
    numpy_data = numpy.arange(7)
    dpnp_data = dpnp.array(numpy_data, device=device)

    expected = numpy.diff(numpy_data, **kwargs)
    result = dpnp.diff(dpnp_data, **kwargs)
    assert_allclose(expected, result)

    expected_queue = dpnp_data.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_clip(device):
    x = dpnp.arange(10, device=device)
    y = dpnp.clip(x, 3, 7)
    assert_sycl_queue_equal(x.sycl_queue, y.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_take(device):
    numpy_data = numpy.arange(5)
    dpnp_data = dpnp.array(numpy_data, device=device)

    dpnp_ind = dpnp.array([0, 2, 4], device=device)
    np_ind = dpnp_ind.asnumpy()

    result = dpnp.take(dpnp_data, dpnp_ind, axis=None)
    expected = numpy.take(numpy_data, np_ind, axis=None)
    assert_allclose(expected, result)

    expected_queue = dpnp_data.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_take_along_axis(data, ind, axis, device):
    dp_data = dpnp.array(data, device=device)
    dp_ind = dpnp.array(ind, device=device)

    result = dpnp.take_along_axis(dp_data, dp_ind, axis=axis)
    expected = numpy.take_along_axis(data, ind, axis=axis)
    assert_allclose(expected, result)

    expected_queue = dp_data.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize("sparse", [True, False], ids=["True", "False"])
def test_indices(device, sparse):
    sycl_queue = dpctl.SyclQueue(device)
    grid = dpnp.indices((2, 3), sparse=sparse, sycl_queue=sycl_queue)
    for dpnp_array in grid:
        assert_sycl_queue_equal(dpnp_array.sycl_queue, sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_nonzero(device):
    a = dpnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
    x = dpnp.nonzero(a)
    for x_el in x:
        assert_sycl_queue_equal(x_el.sycl_queue, a.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize("func", ["mgrid", "ogrid"])
def test_grid(device, func):
    sycl_queue = dpctl.SyclQueue(device)
    x = getattr(dpnp, func)(sycl_queue=sycl_queue)[0:4]
    assert_sycl_queue_equal(x.sycl_queue, sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_where(device):
    a = numpy.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    ia = dpnp.array(a, device=device)

    result = dpnp.where(ia < 4, ia, -1)
    expected = numpy.where(a < 4, a, -1)
    assert_allclose(expected, result)

    expected_queue = ia.sycl_queue
    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
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
    ids=[
        "2D_Matrix_Empty_RHS",
        "2D_Matrix_1D_RHS",
        "3D_Matrix_and_3D_RHS",
    ],
)
def test_solve(matrix, rhs, device):
    a_np = numpy.array(matrix)
    b_np = numpy.array(rhs)

    a_dp = dpnp.array(a_np, device=device)
    b_dp = dpnp.array(b_np, device=device)

    result = dpnp.linalg.solve(a_dp, b_dp)
    expected = numpy.linalg.solve(a_np, b_np)
    assert_dtype_allclose(result, expected)

    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, a_dp.sycl_queue)
    assert_sycl_queue_equal(result_queue, b_dp.sycl_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_slogdet(shape, is_empty, device):
    if is_empty:
        numpy_x = numpy.empty(shape, dtype=dpnp.default_float_type(device))
    else:
        count_elem = numpy.prod(shape)
        numpy_x = numpy.arange(
            1, count_elem + 1, dtype=dpnp.default_float_type(device)
        ).reshape(shape)

    dpnp_x = dpnp.array(numpy_x, device=device)

    sign_result, logdet_result = dpnp.linalg.slogdet(dpnp_x)
    sign_expected, logdet_expected = numpy.linalg.slogdet(numpy_x)
    assert_allclose(logdet_expected, logdet_result, rtol=1e-3, atol=1e-4)
    assert_allclose(sign_expected, sign_result)

    sign_queue = sign_result.sycl_queue
    logdet_queue = logdet_result.sycl_queue

    assert_sycl_queue_equal(sign_queue, dpnp_x.sycl_queue)
    assert_sycl_queue_equal(logdet_queue, dpnp_x.sycl_queue)


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
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_pinv(shape, hermitian, rcond_as_array, device):
    dtype = dpnp.default_float_type(device)
    # Set seed_value=81 to prevent
    # random generation of the input singular matrix
    a_np = generate_random_numpy_array(
        shape, dtype, hermitian=hermitian, seed_value=81
    )
    a_dp = dpnp.array(a_np, device=device)

    if rcond_as_array:
        rcond_np = numpy.array(1e-15)
        rcond_dp = dpnp.array(1e-15, device=device)

        B_result = dpnp.linalg.pinv(a_dp, rcond=rcond_dp, hermitian=hermitian)
        B_expected = numpy.linalg.pinv(
            a_np, rcond=rcond_np, hermitian=hermitian
        )

    else:
        # rcond == 1e-15 by default
        B_result = dpnp.linalg.pinv(a_dp, hermitian=hermitian)
        B_expected = numpy.linalg.pinv(a_np, hermitian=hermitian)

    assert_allclose(B_expected, B_result, rtol=1e-3, atol=1e-4)

    B_queue = B_result.sycl_queue

    assert_sycl_queue_equal(B_queue, a_dp.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_tensorinv(device):
    a_np = numpy.eye(12).reshape(12, 4, 3)
    a_dp = dpnp.array(a_np, device=device)

    result = dpnp.linalg.tensorinv(a_dp, ind=1)
    expected = numpy.linalg.tensorinv(a_np, ind=1)
    assert_dtype_allclose(result, expected)

    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, a_dp.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_tensorsolve(device):
    a_np = numpy.random.randn(3, 2, 6).astype(dpnp.default_float_type())
    b_np = numpy.ones(a_np.shape[:2], dtype=a_np.dtype)

    a_dp = dpnp.array(a_np, device=device)
    b_dp = dpnp.array(b_np, device=device)

    result = dpnp.linalg.tensorsolve(a_dp, b_dp)
    expected = numpy.linalg.tensorsolve(a_np, b_np)
    assert_dtype_allclose(result, expected)

    result_queue = result.sycl_queue

    assert_sycl_queue_equal(result_queue, a_dp.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    ["m", "n", "nrhs"],
    [
        (4, 2, 2),
        (4, 0, 1),
        (4, 2, 0),
        (0, 0, 0),
    ],
)
def test_lstsq(m, n, nrhs, device):
    a_np = numpy.arange(m * n).reshape(m, n).astype(dpnp.default_float_type())
    b_np = numpy.ones((m, nrhs)).astype(dpnp.default_float_type())

    a_dp = dpnp.array(a_np, device=device)
    b_dp = dpnp.array(b_np, device=device)

    result_dp = dpnp.linalg.lstsq(a_dp, b_dp)
    # if rcond is not set, FutureWarning is given.
    # By default Numpy uses None for calculations
    result = numpy.linalg.lstsq(a_np, b_np, rcond=None)

    for param_dp, param_np in zip(result_dp, result):
        assert_dtype_allclose(param_dp, param_np)

    for param_dp in result_dp:
        assert_sycl_queue_equal(param_dp.sycl_queue, a_dp.sycl_queue)
        assert_sycl_queue_equal(param_dp.sycl_queue, b_dp.sycl_queue)


@pytest.mark.parametrize("weights", [None, numpy.arange(7, 12)])
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_bincount(weights, device):
    v = numpy.arange(5)
    w = weights

    iv = dpnp.array(v, device=device)
    iw = None if weights is None else dpnp.array(w, sycl_queue=iv.sycl_queue)

    expected_hist = numpy.bincount(v, weights=w)
    result_hist = dpnp.bincount(iv, weights=iw)
    assert_array_equal(result_hist, expected_hist)

    hist_queue = result_hist.sycl_queue
    assert_sycl_queue_equal(hist_queue, iv.sycl_queue)


@pytest.mark.parametrize("weights", [None, numpy.arange(7, 12)])
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_histogram(weights, device):
    v = numpy.arange(5)
    w = weights

    iv = dpnp.array(v, device=device)
    iw = None if weights is None else dpnp.array(w, sycl_queue=iv.sycl_queue)

    expected_hist, expected_edges = numpy.histogram(v, weights=w)
    result_hist, result_edges = dpnp.histogram(iv, weights=iw)
    assert_array_equal(result_hist, expected_hist)
    assert_dtype_allclose(result_edges, expected_edges)

    hist_queue = result_hist.sycl_queue
    edges_queue = result_edges.sycl_queue
    assert_sycl_queue_equal(hist_queue, iv.sycl_queue)
    assert_sycl_queue_equal(edges_queue, iv.sycl_queue)


@pytest.mark.parametrize("weights", [None, numpy.arange(7, 12)])
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_histogramdd(weights, device):
    v = numpy.arange(5)
    w = weights

    iv = dpnp.array(v, device=device)
    iw = None if weights is None else dpnp.array(w, sycl_queue=iv.sycl_queue)

    expected_hist, expected_edges = numpy.histogramdd(v, weights=w)
    result_hist, result_edges = dpnp.histogramdd(iv, weights=iw)
    assert_array_equal(result_hist, expected_hist)
    for result_edge, expected_edge in zip(result_edges, expected_edges):
        assert_dtype_allclose(result_edge, expected_edge)

    hist_queue = result_hist.sycl_queue
    assert_sycl_queue_equal(hist_queue, iv.sycl_queue)
    for edge in result_edges:
        edges_queue = edge.sycl_queue
        assert_sycl_queue_equal(edges_queue, iv.sycl_queue)


@pytest.mark.parametrize(
    "func", ["tril_indices_from", "triu_indices_from", "diag_indices_from"]
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_tri_diag_indices_from(func, device):
    arr = dpnp.ones((3, 3), device=device)
    res = getattr(dpnp, func)(arr)
    for x in res:
        assert_sycl_queue_equal(x.sycl_queue, arr.sycl_queue)


@pytest.mark.parametrize(
    "func", ["tril_indices", "triu_indices", "diag_indices"]
)
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_tri_diag_indices(func, device):
    sycl_queue = dpctl.SyclQueue(device)
    res = getattr(dpnp, func)(4, sycl_queue=sycl_queue)
    for x in res:
        assert_sycl_queue_equal(x.sycl_queue, sycl_queue)


@pytest.mark.parametrize("mask_func", ["tril", "triu"])
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_mask_indices(mask_func, device):
    sycl_queue = dpctl.SyclQueue(device)
    res = dpnp.mask_indices(4, getattr(dpnp, mask_func), sycl_queue=sycl_queue)
    for x in res:
        assert_sycl_queue_equal(x.sycl_queue, sycl_queue)


@pytest.mark.parametrize("weights", [None, numpy.arange(7, 12)])
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_histogram_bin_edges(weights, device):
    v = numpy.arange(5)
    w = weights

    iv = dpnp.array(v, device=device)
    iw = None if weights is None else dpnp.array(w, sycl_queue=iv.sycl_queue)

    expected_edges = numpy.histogram_bin_edges(v, weights=w)
    result_edges = dpnp.histogram_bin_edges(iv, weights=iw)
    assert_dtype_allclose(result_edges, expected_edges)

    edges_queue = result_edges.sycl_queue
    assert_sycl_queue_equal(edges_queue, iv.sycl_queue)


@pytest.mark.parametrize(
    "device_x",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "device_y",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_astype(device_x, device_y):
    x = dpnp.array([1, 2, 3], dtype="i4", device=device_x)
    y = dpnp.astype(x, dtype="f4")
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)
    sycl_queue = dpctl.SyclQueue(device_y)
    y = dpnp.astype(x, dtype="f4", device=sycl_queue)
    assert_sycl_queue_equal(y.sycl_queue, sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_select(device):
    condlist = [dpnp.array([True, False], device=device)]
    choicelist = [dpnp.array([1, 2], device=device)]
    res = dpnp.select(condlist, choicelist)
    assert_sycl_queue_equal(res.sycl_queue, condlist[0].sycl_queue)


@pytest.mark.parametrize("axis", [None, 0, -1])
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_unique(axis, device):
    a = numpy.array([[1, 1], [2, 3]])
    ia = dpnp.array(a, device=device)

    result = dpnp.unique(ia, True, True, True, axis=axis)
    expected = numpy.unique(a, True, True, True, axis=axis)
    if axis is None and numpy.lib.NumpyVersion(numpy.__version__) < "2.0.1":
        # gh-26961: numpy.unique(..., return_inverse=True, axis=None)
        # returned flatten unique_inverse till 2.0.1 version
        expected = expected[:2] + (expected[2].reshape(a.shape),) + expected[3:]
    for iv, v in zip(result, expected):
        assert_array_equal(iv, v)

        iv_queue = iv.sycl_queue
        assert_sycl_queue_equal(iv_queue, ia.sycl_queue)


@pytest.mark.parametrize("copy", [True, False], ids=["True", "False"])
@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_nan_to_num(copy, device):
    a = dpnp.array([-dpnp.nan, -1, 0, 1, dpnp.nan], device=device)
    result = dpnp.nan_to_num(a, copy=copy)

    assert_sycl_queue_equal(result.sycl_queue, a.sycl_queue)
    assert copy == (result is not a)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    ["to_end", "to_begin"],
    [
        (10, None),
        (None, -10),
        (10, -10),
    ],
)
def test_ediff1d(device, to_end, to_begin):
    data = [1, 3, 5, 7]

    x = dpnp.array(data, device=device)
    if to_end:
        to_end = dpnp.array(to_end, device=device)

    if to_begin:
        to_begin = dpnp.array(to_begin, device=device)

    res = dpnp.ediff1d(x, to_end=to_end, to_begin=to_begin)

    assert_sycl_queue_equal(res.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_unravel_index(device):
    x = dpnp.array(2, device=device)
    result = dpnp.unravel_index(x, shape=(2, 2))
    for res in result:
        assert_sycl_queue_equal(res.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize(
    "device",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_ravel_index(device):
    x = dpnp.array([1, 0], device=device)
    result = dpnp.ravel_multi_index(x, (2, 2))
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize(
    "device_0",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
@pytest.mark.parametrize(
    "device_1",
    valid_devices,
    ids=[device.filter_string for device in valid_devices],
)
def test_ix(device_0, device_1):
    x0 = dpnp.array([0, 1], device=device_0)
    x1 = dpnp.array([2, 4], device=device_1)
    ixgrid = dpnp.ix_(x0, x1)
    assert_sycl_queue_equal(ixgrid[0].sycl_queue, x0.sycl_queue)
    assert_sycl_queue_equal(ixgrid[1].sycl_queue, x1.sycl_queue)
