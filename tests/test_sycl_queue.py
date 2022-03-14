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

available_devices = dpctl.get_devices()

valid_devices = []
for device in available_devices:
    if device.backend.name not in list_of_backend_str:
        pass
    elif device.device_type.name not in list_of_device_type_str:
        pass
    else:
        valid_devices.append(device)


def assert_sycl_queue_equal(result, expected):
    exec_queue = dpctl.utils.get_execution_queue([result, expected])
    assert exec_queue is not None


@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_matmul(device):
    data1 = [[1., 1., 1.], [1., 1., 1.]]
    data2 = [[1., 1.], [1., 1.], [1., 1.]]

    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = numpy.matmul(x1_orig, x2_orig)

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = dpnp.matmul(x1, x2)

    numpy.testing.assert_array_equal(result, expected)

    expected_queue = x1.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device



@pytest.mark.parametrize("func",
                         [])
@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_2in_1out(func, device):
    data1 = [1., 1., 1., 1., 1.]
    data2 = [1., 2., 3., 4., 5.]

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
