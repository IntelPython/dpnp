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
    elif device.backend.name not in list_of_backend_str:
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


def vvsort(val, vec, size, xp):
    val_kwargs = dict()
    if hasattr(val, 'sycl_queue'):
        val_kwargs['sycl_queue'] = getattr(val, "sycl_queue", None)

    vec_kwargs = dict()
    if hasattr(vec, 'sycl_queue'):
        vec_kwargs['sycl_queue'] = getattr(vec, "sycl_queue", None)

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
        pytest.param("arange",
                     [-25.7],
                     {'stop': 10**8, 'step': 15}),
        pytest.param("full",
                     [(2,2)],
                     {'fill_value': 5}),
        pytest.param("eye",
                     [4, 2],
                     {}),
        pytest.param("ones",
                     [(2,2)],
                     {}),
        pytest.param("zeros",
                     [(2,2)],
                     {})
    ])
@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_array_creation(func, arg, kwargs, device):
    numpy_array = getattr(numpy, func)(*arg, **kwargs)

    dpnp_kwargs = dict(kwargs)
    dpnp_kwargs['device'] = device
    dpnp_array = getattr(dpnp, func)(*arg, **dpnp_kwargs)

    numpy.testing.assert_array_equal(numpy_array, dpnp_array)
    assert dpnp_array.sycl_device == device


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_empty(device):
    dpnp_array = dpnp.empty((2, 2), device=device)
    assert dpnp_array.sycl_device == device


@pytest.mark.parametrize("device_x",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
@pytest.mark.parametrize("device_y",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_empty_like(device_x, device_y):
    x = dpnp.ndarray([1, 2, 3], device=device_x)
    y = dpnp.empty_like(x)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)
    y = dpnp.empty_like(x, device=device_y)
    assert_sycl_queue_equal(y.sycl_queue, x.to_device(device_y).sycl_queue)


@pytest.mark.parametrize(
    "func, kwargs",
    [
        pytest.param("full_like",
                     {'fill_value': 5}),
        pytest.param("ones_like",
                     {}),
        pytest.param("zeros_like",
                     {})
    ])
@pytest.mark.parametrize("device_x",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
@pytest.mark.parametrize("device_y",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_array_creation_like(func, kwargs, device_x, device_y):
    x_orig = numpy.ndarray([1, 2, 3])
    y_orig = getattr(numpy, func)(x_orig, **kwargs)

    x = dpnp.ndarray([1, 2, 3], device=device_x)

    y = getattr(dpnp, func)(x, **kwargs)
    numpy.testing.assert_array_equal(y_orig, y)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)

    dpnp_kwargs = dict(kwargs)
    dpnp_kwargs['device'] = device_y
    
    y = getattr(dpnp, func)(x, **dpnp_kwargs)
    numpy.testing.assert_array_equal(y_orig, y)
    assert_sycl_queue_equal(y.sycl_queue, x.to_device(device_y).sycl_queue)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
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
        pytest.param("matmul",
                     [[1., 0.], [0., 1.]],
                     [[4., 1.], [1., 2.]]),
        pytest.param("maximum",
                     [2., 3., 4.],
                     [1., 5., 2.]),
        pytest.param("minimum",
                     [2., 3., 4.],
                     [1., 5., 2.]),
        pytest.param("multiply",
                     [0., 1., 2., 3., 4., 5., 6., 7., 8.],
                     [0., 1., 2., 0., 1., 2., 0., 1., 2.]),
        pytest.param("outer",
                     [0., 1., 2., 3., 4., 5.],
                     [0., 1., 2., 0.]),
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
def test_2in_1out(func, data1, data2, device):
    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = getattr(numpy, func)(x1_orig, x2_orig)

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)(x1, x2)

    numpy.testing.assert_array_equal(result, expected)

    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, x2.sycl_queue)


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


@pytest.mark.parametrize("usm_type",
                         ["host", "device", "shared"])
@pytest.mark.parametrize("size",
                         [None, (), 3, (2, 1), (4, 2, 5)],
                         ids=['None', '()', '3', '(2,1)', '(4,2,5)'])
def test_uniform(usm_type, size):
    low = 1.0
    high = 2.0
    res = dpnp.random.uniform(low, high, size=size, usm_type=usm_type)

    assert usm_type == res.usm_type


@pytest.mark.parametrize("usm_type",
                         ["host", "device", "shared"])
@pytest.mark.parametrize("seed",
                         [None, (), 123, (12, 58), (147, 56, 896), [1, 654, 78]],
                         ids=['None', '()', '123', '(12,58)', '(147,56,896)', '[1,654,78]'])
def test_rs_uniform(usm_type, seed):
    seed = 123
    sycl_queue = dpctl.SyclQueue()
    low = 1.0
    high = 2.0
    rs = dpnp.random.RandomState(seed, sycl_queue=sycl_queue)
    res = rs.uniform(low, high, usm_type=usm_type)

    assert usm_type == res.usm_type

    res_sycl_queue = res.get_array().sycl_queue
    assert_sycl_queue_equal(res_sycl_queue, sycl_queue)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
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


@pytest.mark.parametrize("type", ['complex128'])
@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_fft(type, device):
    data = numpy.arange(100, dtype=numpy.dtype(type))

    dpnp_data = dpnp.array(data, device=device)

    expected = numpy.fft.fft(data)
    result = dpnp.fft.fft(dpnp_data)

    numpy.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-7)

    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize("type", ['float32'])
@pytest.mark.parametrize("shape", [(8,8)])
@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_fft_rfft(type, shape, device):
    np_data = numpy.arange(64, dtype=numpy.dtype(type)).reshape(shape)
    dpnp_data = dpnp.array(np_data, device=device)

    np_res = numpy.fft.rfft(np_data)
    dpnp_res = dpnp.fft.rfft(dpnp_data)

    numpy.testing.assert_allclose(dpnp_res, np_res, rtol=1e-4, atol=1e-7)
    assert dpnp_res.dtype == np_res.dtype

    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = dpnp_res.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_cholesky(device):
    data = [[[1., -2.], [2., 5.]], [[1., -2.], [2., 5.]]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.cholesky(dpnp_data)
    expected = numpy.linalg.cholesky(numpy_data)
    numpy.testing.assert_array_equal(expected, result)

    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_det(device):
    data = [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.det(dpnp_data)
    expected = numpy.linalg.det(numpy_data)
    numpy.testing.assert_allclose(expected, result)

    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_eig(device):
    if device.device_type != dpctl.device_type.gpu:
        pytest.skip("eig function doesn\'t work on CPU: https://github.com/IntelPython/dpnp/issues/1005")

    size = 4
    a = numpy.arange(size * size, dtype='float64').reshape((size, size))
    symm_orig = numpy.tril(a) + numpy.tril(a, -1).T + numpy.diag(numpy.full((size,), size * size, dtype='float64'))
    numpy_data = symm_orig
    dpnp_symm_orig = dpnp.array(numpy_data, device=device)
    dpnp_data = dpnp_symm_orig

    dpnp_val, dpnp_vec = dpnp.linalg.eig(dpnp_data)
    numpy_val, numpy_vec = numpy.linalg.eig(numpy_data)

    # DPNP sort val/vec by abs value
    vvsort(dpnp_val, dpnp_vec, size, dpnp)

    # NP sort val/vec by abs value
    vvsort(numpy_val, numpy_vec, size, numpy)

    # NP change sign of vectors
    for i in range(numpy_vec.shape[1]):
        if numpy_vec[0, i] * dpnp_vec[0, i] < 0:
            numpy_vec[:, i] = -numpy_vec[:, i]

    numpy.testing.assert_allclose(dpnp_val, numpy_val, rtol=1e-05, atol=1e-05)
    numpy.testing.assert_allclose(dpnp_vec, numpy_vec, rtol=1e-05, atol=1e-05)

    assert (dpnp_val.dtype == numpy_val.dtype)
    assert (dpnp_vec.dtype == numpy_vec.dtype)
    assert (dpnp_val.shape == numpy_val.shape)
    assert (dpnp_vec.shape == numpy_vec.shape)

    expected_queue = dpnp_data.get_array().sycl_queue
    dpnp_val_queue = dpnp_val.get_array().sycl_queue
    dpnp_vec_queue = dpnp_vec.get_array().sycl_queue

    # compare queue and device    
    assert_sycl_queue_equal(dpnp_val_queue, expected_queue)
    assert_sycl_queue_equal(dpnp_vec_queue, expected_queue)


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_eigvals(device):
    if device.device_type != dpctl.device_type.gpu:
        pytest.skip("eigvals function doesn\'t work on CPU: https://github.com/IntelPython/dpnp/issues/1005")

    data = [[0, 0], [0, 0]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.eigvals(dpnp_data)
    expected = numpy.linalg.eigvals(numpy_data)
    numpy.testing.assert_allclose(expected, result, atol=0.5)

    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_inv(device):
    data = [[1., 2.], [3., 4.]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.inv(dpnp_data)
    expected = numpy.linalg.inv(numpy_data)
    numpy.testing.assert_allclose(expected, result)

    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_matrix_rank(device):
    data = [[0, 0], [0, 0]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.matrix_rank(dpnp_data)
    expected = numpy.linalg.matrix_rank(numpy_data)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_qr(device):
    tol = 1e-11
    data = [[1,2,3], [1,2,3]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    np_q, np_r = numpy.linalg.qr(numpy_data, "reduced")
    dpnp_q, dpnp_r = dpnp.linalg.qr(dpnp_data, "reduced")

    assert (dpnp_q.dtype == np_q.dtype)
    assert (dpnp_r.dtype == np_r.dtype)
    assert (dpnp_q.shape == np_q.shape)
    assert (dpnp_r.shape == np_r.shape)

    numpy.testing.assert_allclose(dpnp_q, np_q, rtol=tol, atol=tol)
    numpy.testing.assert_allclose(dpnp_r, np_r, rtol=tol, atol=tol)

    expected_queue = dpnp_data.get_array().sycl_queue
    dpnp_q_queue = dpnp_q.get_array().sycl_queue
    dpnp_r_queue = dpnp_r.get_array().sycl_queue

    # compare queue and device
    assert_sycl_queue_equal(dpnp_q_queue, expected_queue)
    assert_sycl_queue_equal(dpnp_r_queue, expected_queue)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("device",
                        valid_devices,
                        ids=[device.filter_string for device in valid_devices])
def test_svd(device):
    tol = 1e-12
    shape = (2,2)
    numpy_data = numpy.arange(shape[0] * shape[1]).reshape(shape)
    dpnp_data = dpnp.arange(shape[0] * shape[1]).reshape(shape)
    np_u, np_s, np_vt = numpy.linalg.svd(numpy_data)
    dpnp_u, dpnp_s, dpnp_vt = dpnp.linalg.svd(dpnp_data)

    assert (dpnp_u.dtype == np_u.dtype)
    assert (dpnp_s.dtype == np_s.dtype)
    assert (dpnp_vt.dtype == np_vt.dtype)
    assert (dpnp_u.shape == np_u.shape)
    assert (dpnp_s.shape == np_s.shape)
    assert (dpnp_vt.shape == np_vt.shape)

    # check decomposition
    dpnp_diag_s = dpnp.zeros(shape, dtype=dpnp_s.dtype)
    for i in range(dpnp_s.size):
        dpnp_diag_s[i, i] = dpnp_s[i]

    # check decomposition
    numpy.testing.assert_allclose(dpnp_data, dpnp.dot(dpnp_u, dpnp.dot(dpnp_diag_s, dpnp_vt)), rtol=tol, atol=tol)

    for i in range(min(shape[0], shape[1])):
        if np_u[0, i] * dpnp_u[0, i] < 0:
            np_u[:, i] = -np_u[:, i]
            np_vt[i, :] = -np_vt[i, :]

    # compare vectors for non-zero values
    for i in range(numpy.count_nonzero(np_s > tol)):
        numpy.testing.assert_allclose(dpnp.asnumpy(dpnp_u)[:, i], np_u[:, i], rtol=tol, atol=tol)
        numpy.testing.assert_allclose(dpnp.asnumpy(dpnp_vt)[i, :], np_vt[i, :], rtol=tol, atol=tol)

    expected_queue = dpnp_data.get_array().sycl_queue
    dpnp_u_queue = dpnp_u.get_array().sycl_queue
    dpnp_s_queue = dpnp_s.get_array().sycl_queue
    dpnp_vt_queue = dpnp_vt.get_array().sycl_queue

    # compare queue and device
    assert_sycl_queue_equal(dpnp_u_queue, expected_queue)
    assert_sycl_queue_equal(dpnp_s_queue, expected_queue)
    assert_sycl_queue_equal(dpnp_vt_queue, expected_queue)


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
