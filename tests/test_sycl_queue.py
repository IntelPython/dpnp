import pytest

import dpnp
import dpctl
import numpy


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

available_devices = [d for d in dpctl.get_devices() if not d.has_aspect_host]

valid_devices = []
for device in available_devices:
    if device.default_selector_score < 0:
        pass
    if device.backend.name not in list_of_backend_str:
        pass
    elif device.device_type.name not in list_of_device_type_str:
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


@pytest.mark.parametrize(
    "func,data",
    [
        pytest.param("abs",
                     [-1.2, 1.2]),
        pytest.param("ceil",
                     [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
        pytest.param("conjugate",
                     [[1.+1.j, 0.], [0., 1.+1.j]]),
        pytest.param("copy",
                     [1., 2., 3.]),
        pytest.param("cumprod",
                     [[1., 2., 3.], [4., 5., 6.]]),
        pytest.param("cumsum",
                     [[1., 2., 3.], [4., 5., 6.]]),
        pytest.param("diff",
                     [1., 2., 4., 7., 0.]),
        pytest.param("ediff1d",
                     [1., 2., 4., 7., 0.]),
        pytest.param("fabs",
                     [-1.2, 1.2]),
        pytest.param("floor",
                     [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
        pytest.param("gradient",
                     [1., 2., 4., 7., 11., 16.]),
        pytest.param("nancumprod",
                     [1., dpnp.nan]),
        pytest.param("nancumsum",
                     [1., dpnp.nan]),
        pytest.param("nanprod",
                     [1., dpnp.nan]),
        pytest.param("nansum",
                     [1., dpnp.nan]),
        pytest.param("negative",
                     [1., -1.]),
        pytest.param("prod",
                     [1., 2.]),
        pytest.param("sign",
                     [-5., 4.5]),
        pytest.param("sum",
                     [1., 2.]),
        pytest.param("trapz",
                     [[0., 1., 2.], [3., 4., 5.]]),
        pytest.param("trunc",
                     [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
    ],
)
@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_1in_1out(func, data, device):
    x_orig = numpy.array(data)
    expected = getattr(numpy, func)(x_orig)

    x = dpnp.array(data, device=device)
    result = getattr(dpnp, func)(x)

    numpy.testing.assert_array_equal(result, expected)

    expected_queue = x.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize(
    "func,data1,data2",
    [
        pytest.param("add",
                     [0., 1., 2., 3., 4., 5., 6., 7., 8.],
                     [0., 1., 2., 0., 1., 2., 0., 1., 2.]),
        pytest.param("copysign",
                     [0., 1., 2.],
                     [-1., 0., 1.]),
        pytest.param("cross",
                     [1., 2., 3.],
                     [4., 5., 6.]),
        pytest.param("divide",
                     [0., 1., 2., 3., 4.],
                     [4., 4., 4., 4., 4.]),
        pytest.param("floor_divide",
                     [1., 2., 3., 4.],
                     [2.5, 2.5, 2.5, 2.5]),
        pytest.param("fmod",
                     [-3., -2., -1., 1., 2., 3.],
                     [2., 2., 2., 2., 2., 2.]),
        pytest.param("maximum",
                     [2., 3., 4.],
                     [1., 5., 2.]),
        pytest.param("minimum",
                     [2., 3., 4.],
                     [1., 5., 2.]),
        pytest.param("multiply",
                     [0., 1., 2., 3., 4., 5., 6., 7., 8.],
                     [0., 1., 2., 0., 1., 2., 0., 1., 2.]),
        pytest.param("power",
                     [0., 1., 2., 3., 4., 5.],
                     [1., 2., 3., 3., 2., 1.]),
        pytest.param("remainder",
                     [0., 1., 2., 3., 4., 5., 6.],
                     [5., 5., 5., 5., 5., 5., 5.]),
        pytest.param("subtract",
                     [0., 1., 2., 3., 4., 5., 6., 7., 8.],
                     [0., 1., 2., 0., 1., 2., 0., 1., 2.]),
        pytest.param("matmul",
                     [[1., 0.], [0., 1.]],
                     [[4., 1.], [1., 2.]]),
    ],
)
@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_2in_1out(func, data1, data2, device):
    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = getattr(numpy, func)(x1_orig, x2_orig)

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)(x1, x2)

    numpy.testing.assert_array_equal(result, expected)

    expected_queue = x1.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize(
    "func,data1,data2",
    [
        pytest.param("add",
                     [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
                     [0., 1., 2.]),
        pytest.param("divide",
                     [0., 1., 2., 3., 4.],
                     [4.]),
        pytest.param("floor_divide",
                     [1., 2., 3., 4.],
                     [2.5]),
        pytest.param("fmod",
                     [-3., -2., -1., 1., 2., 3.],
                     [2.]),
        pytest.param("multiply",
                     [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
                     [0., 1., 2.]),
        pytest.param("remainder",
                     [0., 1., 2., 3., 4., 5., 6.],
                     [5.]),
        pytest.param("subtract",
                     [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
                     [0., 1., 2.]),
    ],
)
@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_broadcasting(func, data1, data2, device):
    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = getattr(numpy, func)(x1_orig, x2_orig)

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)(x1, x2)

    numpy.testing.assert_array_equal(result, expected)

    expected_queue = x1.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize(
    "func,data1,data2",
    [
        pytest.param("add",
                     [0., 1., 2., 3., 4., 5., 6., 7., 8.],
                     [0., 1., 2., 0., 1., 2., 0., 1., 2.]),
        pytest.param("copysign",
                     [0., 1., 2.],
                     [-1., 0., 1.]),
        pytest.param("divide",
                     [0., 1., 2., 3., 4.],
                     [4., 4., 4., 4., 4.]),
        pytest.param("floor_divide",
                     [1., 2., 3., 4.],
                     [2.5, 2.5, 2.5, 2.5]),
        pytest.param("fmod",
                     [-3., -2., -1., 1., 2., 3.],
                     [2., 2., 2., 2., 2., 2.]),
        pytest.param("maximum",
                     [2., 3., 4.],
                     [1., 5., 2.]),
        pytest.param("minimum",
                     [2., 3., 4.],
                     [1., 5., 2.]),
        pytest.param("multiply",
                     [0., 1., 2., 3., 4., 5., 6., 7., 8.],
                     [0., 1., 2., 0., 1., 2., 0., 1., 2.]),
        pytest.param("power",
                     [0., 1., 2., 3., 4., 5.],
                     [1., 2., 3., 3., 2., 1.]),
        pytest.param("remainder",
                     [0., 1., 2., 3., 4., 5., 6.],
                     [5., 5., 5., 5., 5., 5., 5.]),
        pytest.param("subtract",
                     [0., 1., 2., 3., 4., 5., 6., 7., 8.],
                     [0., 1., 2., 0., 1., 2., 0., 1., 2.]),
    ],
)
@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_out(func, data1, data2, device):
    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = numpy.empty(x1_orig.size)
    numpy.add(x1_orig, x2_orig, out=expected)

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = dpnp.empty(x1.size, device=device)
    dpnp.add(x1, x2, out=result)

    numpy.testing.assert_array_equal(result, expected)

    expected_queue = x1.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_modf(device):
    data = [0, 3.5]

    x_orig = numpy.array(data)
    expected1, expected2 = numpy.modf(x_orig)

    x = dpnp.array(data, device=device)
    result1, result2 = dpnp.modf(x)

    numpy.testing.assert_array_equal(result1, expected1)
    numpy.testing.assert_array_equal(result2, expected2)

    expected_queue = x.get_array().sycl_queue
    result1_queue = result1.get_array().sycl_queue
    result2_queue = result2.get_array().sycl_queue

    assert_sycl_queue_equal(result1_queue, expected_queue)
    assert_sycl_queue_equal(result2_queue, expected_queue)

    assert result1_queue.sycl_device == expected_queue.sycl_device
    assert result2_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("device_from",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
@pytest.mark.parametrize("device_to",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_to_device(device_from, device_to):
    data = [1., 1., 1., 1., 1.]

    x = dpnp.array(data, device=device_from)
    y = x.to_device(device_to)

    assert y.get_array().sycl_device == device_to


@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
@pytest.mark.parametrize("func",
                         ["array", "asarray"])
@pytest.mark.parametrize("device_param",
                         ["", "None", "sycl_device"],
                         ids=['Empty', 'None', "device"])
@pytest.mark.parametrize("queue_param",
                         ["", "None", "sycl_queue"],
                         ids=['Empty', 'None', "queue"])
def test_array_copy(device, func, device_param, queue_param):
    data = numpy.ones(100)
    dpnp_data = getattr(dpnp, func)(data, device=device)

    kwargs_items = {'device': device_param, 'sycl_queue': queue_param}.items()
    kwargs = {k: getattr(dpnp_data, v, None) for k,v in kwargs_items if v != ""}

    result = dpnp.array(dpnp_data, **kwargs)

    assert_sycl_queue_equal(result.sycl_queue, dpnp_data.sycl_queue)
