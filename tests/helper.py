from sys import platform

import dpctl
import numpy
from numpy.testing import assert_allclose, assert_array_equal

import dpnp


def assert_dtype_allclose(
    dpnp_arr,
    numpy_arr,
    check_type=True,
    check_only_type_kind=False,
    factor=8,
):
    """
    Assert DPNP and NumPy array based on maximum dtype resolution of input arrays
    for floating and complex types.
    For other dtypes the assertion is based on exact matching of the arrays.
    When 'check_type' is True (default), the function asserts:
    - Equal dtypes for exact types.
    For inexact types:
      - If the numpy array's dtype is `numpy.float16`, checks if the device
        of the `dpnp_arr` supports 64-bit precision floating point operations.
        If supported, asserts equal dtypes.
        Otherwise, asserts equal type kinds.
      - For other inexact types, asserts equal dtypes if the device of the `dpnp_arr`
        supports 64-bit precision floating point operations or if the numpy array's inexact
        dtype is not a double precision type.
        Otherwise, asserts equal type kinds.
    The 'check_only_type_kind' parameter (False by default) asserts only equal type kinds
    for all data types supported by DPNP when set to True.
    It is effective only when 'check_type' is also set to True.
    The parameter `factor` scales the resolution used for comparing the arrays.

    """

    list_64bit_types = [numpy.float64, numpy.complex128]
    is_inexact = lambda x: dpnp.issubdtype(x.dtype, dpnp.inexact)
    if is_inexact(dpnp_arr) or is_inexact(numpy_arr):
        tol_dpnp = (
            dpnp.finfo(dpnp_arr).resolution
            if is_inexact(dpnp_arr)
            else -dpnp.inf
        )
        tol_numpy = (
            numpy.finfo(numpy_arr.dtype).resolution
            if is_inexact(numpy_arr)
            else -dpnp.inf
        )
        tol = factor * max(tol_dpnp, tol_numpy)
        assert_allclose(dpnp_arr.asnumpy(), numpy_arr, atol=tol, rtol=tol)
        if check_type:
            numpy_arr_dtype = numpy_arr.dtype
            dpnp_arr_dtype = dpnp_arr.dtype
            dpnp_arr_dev = dpnp_arr.sycl_device

            if check_only_type_kind:
                assert dpnp_arr_dtype.kind == numpy_arr_dtype.kind
            else:
                is_np_arr_f2 = numpy_arr_dtype == numpy.float16

                if is_np_arr_f2:
                    if has_support_aspect16(dpnp_arr_dev):
                        assert dpnp_arr_dtype == numpy_arr_dtype
                elif (
                    numpy_arr_dtype not in list_64bit_types
                    or has_support_aspect64(dpnp_arr_dev)
                ):
                    assert dpnp_arr_dtype == numpy_arr_dtype
                else:
                    assert dpnp_arr_dtype.kind == numpy_arr_dtype.kind
    else:
        assert_array_equal(dpnp_arr.asnumpy(), numpy_arr)
        if check_type:
            if check_only_type_kind:
                assert dpnp_arr.dtype.kind == numpy_arr.dtype.kind
            else:
                assert dpnp_arr.dtype == numpy_arr.dtype


def get_integer_dtypes():
    """
    Build a list of integer types supported by DPNP.
    """

    return [dpnp.int32, dpnp.int64]


def get_integer_dtypes():
    """
    Build a list of integer types supported by DPNP.
    """

    return [dpnp.int32, dpnp.int64]


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
    dtypes.extend(get_integer_dtypes())

    # add floating types
    dtypes.extend(get_float_dtypes(no_float16=no_float16, device=dev))

    # add complex types
    if not no_complex:
        dtypes.extend(get_complex_dtypes(device=dev))

    # add None value to validate a default dtype
    if not no_none:
        dtypes.append(None)
    return dtypes


def get_symm_herm_numpy_array(shape, dtype=None):
    """
    Generates a real symmetric or a complex Hermitian numpy array of
    the specified shape and data type.

    Note:
    For arrays with more than 2 dimensions, it ensures symmetry(or Hermitian property
    for complex data type) for each sub-array.

    """

    numpy.random.seed(81)
    a = numpy.random.randn(*shape).astype(dtype)
    if numpy.issubdtype(a.dtype, numpy.complexfloating):
        a += 1j * numpy.random.randn(*shape)

    if a.size > 0:
        if a.ndim > 2:
            for i in range(a.shape[0]):
                a[i] = numpy.conj(a[i].T) @ a[i]
        else:
            a = numpy.conj(a.T) @ a
    return a


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
