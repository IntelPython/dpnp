import dpctl
import dpnp
import pytest


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


def skip_or_check_if_dtype_not_supported(dtype, device=None, check_dtype=False):
    """
    The function to check input type supported in DPNP based on the device capabilities.
    """

    dev = dpctl.select_default_device() if device is None else device
    dev_has_dp = dev.has_aspect_fp64
    if dtype is dpnp.float64 and dev_has_dp is False:
        if check_dtype:
            return False
        else:
            pytest.skip(
                f"{dev.name} does not support double precision floating point types"
            )
    dev_has_hp = dev.has_aspect_fp16
    if dtype is dpnp.complex128 and dev_has_hp is False:
        if check_dtype:
            return False
        else:
            pytest.skip(
                f"{dev.name} does not support double precision floating point types"
            )

    return True
