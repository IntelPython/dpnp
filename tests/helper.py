from sys import platform

import dpctl
import dpnp


def get_all_dtypes(no_bool=False,
                   no_float16=True,
                   no_complex=False,
                   no_none=False,
                   device=None):
    """
    Build a list of types supported by DPNP based on input flags and device capabilities.
    """

    dev = dpctl.select_default_device() if device is None else device

    # add boolean type
    dtypes = [dpnp.bool] if not no_bool else []

    # add integer types
    dtypes.extend([dpnp.int32, dpnp.int64])

    # add floating types
    if not no_float16 and dev.has_aspect_fp16:
        dtypes.append(dpnp.float16)

    dtypes.append(dpnp.float32)
    if dev.has_aspect_fp64:
        dtypes.append(dpnp.float64)

    # add complex types
    if not no_complex:
        dtypes.append(dpnp.complex64)
        if dev.has_aspect_fp64:
            dtypes.append(dpnp.complex128)

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
    return platform.startswith('win')
