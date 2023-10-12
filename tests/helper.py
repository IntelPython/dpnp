from sys import platform

import dpctl
import numpy
from numpy.testing import assert_allclose, assert_array_equal

import dpnp


def assert_dtype_allclose(dpnp_arr, numpy_arr, check_type=True):
    """
    Assert DPNP and NumPy array based on maximum dtype resolution of input arrays
    for floating and complex types.
    For other dtypes the assertion is based on exact matching of the arrays.

    """

    is_inexact = lambda x: dpnp.issubdtype(x.dtype, dpnp.inexact)
    if is_inexact(dpnp_arr) or is_inexact(numpy_arr):
        tol = 8 * max(
            dpnp.finfo(dpnp_arr).resolution,
            numpy.finfo(numpy_arr.dtype).resolution,
        )
        assert_allclose(dpnp_arr.asnumpy(), numpy_arr, atol=tol, rtol=tol)
    else:
        assert_array_equal(dpnp_arr.asnumpy(), numpy_arr)
        if check_type:
            assert dpnp_arr.dtype == numpy_arr.dtype


def get_complex_dtypes(device=None):
    """
    Build a list of complex types supported by DPNP based on device capabilities.
    """

    dev = dpctl.select_default_device() if device is None else device

    # add complex types
    dtypes = [dpnp.complex64]
    if dev.has_aspect_fp64:
        dtypes.append(dpnp.complex128)
    return dtypes


def get_float_dtypes(no_float16=True, device=None):
    """
    Build a list of floating types supported by DPNP based on device capabilities.
    """

    dev = dpctl.select_default_device() if device is None else device

    # add floating types
    dtypes = []
    if not no_float16 and dev.has_aspect_fp16:
        dtypes.append(dpnp.float16)

    dtypes.append(dpnp.float32)
    if dev.has_aspect_fp64:
        dtypes.append(dpnp.float64)
    return dtypes


def get_float_complex_dtypes(no_float16=True, device=None):
    """
    Build a list of floating and complex types supported by DPNP based on device capabilities.
    """

    dtypes = get_float_dtypes(no_float16, device)
    dtypes.extend(get_complex_dtypes(device))
    return dtypes


def get_all_dtypes(
    no_bool=False, no_float16=True, no_complex=False, no_none=False, device=None
):
    """
    Build a list of types supported by DPNP based on input flags and device capabilities.
    """

    dev = dpctl.select_default_device() if device is None else device

    # add boolean type
    dtypes = [dpnp.bool] if not no_bool else []

    # add integer types
    dtypes.extend([dpnp.int32, dpnp.int64])

    # add floating types
    dtypes.extend(get_float_dtypes(no_float16=no_float16, device=dev))

    # add complex types
    if not no_complex:
        dtypes.extend(get_complex_dtypes(device=dev))

    # add None value to validate a default dtype
    if not no_none:
        dtypes.append(None)
    return dtypes


def is_cpu_device(device=None):
    """
    Return True if a test is running on CPU device, False otherwise.
    """
    dev = dpctl.select_default_device() if device is None else device
    return dev.has_aspect_cpu


def is_win_platform():
    """
    Return True if a test is runing on Windows OS, False otherwise.
    """
    return platform.startswith("win")


def has_support_aspect16(device=None):
    """
    Return True if the device supports 16-bit precision floating point operations,
    False otherwise.
    """
    dev = dpctl.select_default_device() if device is None else device
    return dev.has_aspect_fp16


def has_support_aspect64(device=None):
    """
    Return True if the device supports 64-bit precision floating point operations,
    False otherwise.
    """
    dev = dpctl.select_default_device() if device is None else device
    return dev.has_aspect_fp64
