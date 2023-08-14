from itertools import product

import dpctl
import dpctl.tensor as dpt
import pytest

import dpnp
from dpnp.backend.extensions.sycl_ext import _sycl_ext_impl

all_devices = [
    device for device in dpctl.get_devices() if device.is_cpu or device.is_gpu
]
sum_supported_input_dtypes = [
    dpt.dtype("i1"),
    dpt.dtype("u1"),
    dpt.dtype("i2"),
    dpt.dtype("u2"),
    dpt.dtype("i4"),
    dpt.dtype("u4"),
    dpt.dtype("i8"),
    dpt.dtype("u8"),
    dpt.float32,
    dpt.float64,
]
sum_supported_output_dtypes = [
    dpt.dtype("i4"),
    dpt.dtype("u4"),
    dpt.dtype("i8"),
    dpt.dtype("u8"),
    dpt.float32,
    dpt.float64,
]

mean_supported_output_dtypes = [dpt.float32, dpt.float64]
sum_without_mean_supported_dtypes = list(
    set(sum_supported_output_dtypes) - set(mean_supported_output_dtypes)
)

sum_unsupported_input_dtypes = [
    dpt.bool,
    dpt.float16,
    dpt.complex64,
    dpt.complex128,
]
sum_unsupported_output_dtypes = [
    dpt.bool,
    dpt.dtype("i1"),
    dpt.dtype("u1"),
    dpt.dtype("i2"),
    dpt.dtype("u2"),
    dpt.float16,
    dpt.complex64,
    dpt.complex128,
]

mean_unsupported_output_dtypes = [
    dpt.bool,
    dpt.dtype("i1"),
    dpt.dtype("u1"),
    dpt.dtype("i2"),
    dpt.dtype("u2"),
    dpt.dtype("i4"),
    dpt.dtype("u4"),
    dpt.dtype("i8"),
    dpt.dtype("u8"),
    dpt.float16,
    dpt.complex64,
    dpt.complex128,
]


sum_only = [_sycl_ext_impl._get_sum_over_axis_0]
mean_only = [_sycl_ext_impl._get_mean_over_axis_0]
mean_sum = sum_only + mean_only


def supported_by_device(device, typ):
    if typ == dpt.float64 or typ == dpt.complex128:
        return device.has_aspect_fp64

    if typ == dpt.float16:
        return device.has_aspect_fp16

    return True


def skip_unsupported(device, typ):
    if not supported_by_device(device, typ):
        pytest.skip(f"{typ} type is not supported by {device}")


@pytest.mark.parametrize(
    "func, device, input_type, output_type",
    product(
        mean_sum,
        all_devices,
        sum_supported_input_dtypes,
        mean_supported_output_dtypes,
    ),
)
def test_mean_sum_over_axis_0_supported_types(
    func, device, input_type, output_type
):
    skip_unsupported(device, input_type)
    skip_unsupported(device, output_type)

    height = 20
    width = 10

    input = dpt.empty((height, width), dtype=input_type, device=device)
    output = dpt.empty(width, dtype=output_type, device=device)

    assert func(input, output) is not None


@pytest.mark.parametrize(
    "func, device, input_type, output_type",
    product(
        sum_only,
        all_devices,
        sum_supported_input_dtypes,
        sum_without_mean_supported_dtypes,
    ),
)
def test_sum_over_axis_0_supported_types(func, device, input_type, output_type):
    skip_unsupported(device, input_type)
    skip_unsupported(device, output_type)

    height = 20
    width = 10

    input = dpt.empty((height, width), dtype=input_type, device=device)
    output = dpt.empty(width, dtype=output_type, device=device)

    assert func(input, output) is not None


@pytest.mark.parametrize(
    "func, device, input_type, output_type",
    product(mean_sum, all_devices, sum_unsupported_input_dtypes, [dpt.float32]),
)
def test_mean_sum_over_axis_0_unsupported_in_types(
    func, device, input_type, output_type
):
    skip_unsupported(device, input_type)
    skip_unsupported(device, output_type)

    height = 1
    width = 1

    input = dpt.empty((height, width), dtype=input_type, device=device)
    output = dpt.empty(width, dtype=output_type, device=device)

    assert func(input, output) is None


@pytest.mark.parametrize(
    "func, device, input_type, output_type",
    product(
        sum_only, all_devices, [dpt.float32], sum_unsupported_output_dtypes
    ),
)
def test_sum_over_axis_0_unsupported_out_types(
    func, device, input_type, output_type
):
    skip_unsupported(device, input_type)
    skip_unsupported(device, output_type)

    height = 1
    width = 1

    input = dpt.empty((height, width), dtype=input_type, device=device)
    output = dpt.empty(width, dtype=output_type, device=device)

    assert func(input, output) is None


@pytest.mark.parametrize(
    "func, device, input_type, output_type",
    product(
        mean_only, all_devices, [dpt.float32], mean_unsupported_output_dtypes
    ),
)
def test_mean_over_axis_0_unsupported_out_types(
    func, device, input_type, output_type
):
    skip_unsupported(device, input_type)
    skip_unsupported(device, output_type)

    height = 1
    width = 1

    input = dpt.empty((height, width), dtype=input_type, device=device)
    output = dpt.empty(width, dtype=output_type, device=device)

    assert func(input, output) is None


@pytest.mark.parametrize(
    "func, device, input_type, output_type",
    product(mean_sum, all_devices, [dpt.float32], [dpt.float32]),
)
def test_mean_sum_over_axis_0_f_contig_input(
    func, device, input_type, output_type
):
    skip_unsupported(device, input_type)
    skip_unsupported(device, output_type)

    height = 20
    width = 10

    input = dpt.empty((height, width), dtype=input_type, device=device).T
    output = dpt.empty(width, dtype=output_type, device=device)

    assert func(input, output) is None


@pytest.mark.parametrize(
    "func, device, input_type, output_type",
    product(mean_sum, all_devices, [dpt.float32], [dpt.float32]),
)
def test_mean_sum_over_axis_0_f_contig_output(
    func, device, input_type, output_type
):
    skip_unsupported(device, input_type)
    skip_unsupported(device, output_type)

    height = 1
    width = 10

    input = dpt.empty((height, width), dtype=input_type, device=device)
    output = dpt.empty(width * 2, dtype=output_type, device=device)[::2]

    assert func(input, output) is None


@pytest.mark.parametrize(
    "func, device, input_type, output_type",
    product(mean_sum, all_devices, [dpt.float32], [dpt.float32, dpt.float64]),
)
def test_mean_sum_over_axis_0_big_output(func, device, input_type, output_type):
    skip_unsupported(device, input_type)
    skip_unsupported(device, output_type)

    local_mem_size = device.local_mem_size
    height = 1
    width = 1 + local_mem_size // output_type.itemsize

    input = dpt.empty((height, width), dtype=input_type, device=device)
    output = dpt.empty(width, dtype=output_type, device=device)

    assert func(input, output) is None
