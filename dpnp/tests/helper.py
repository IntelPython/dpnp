from sys import platform

import dpctl
import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpnp

from . import config


def _assert_dtype(a_dt, b_dt, check_only_type_kind=False):
    if check_only_type_kind:
        assert a_dt.kind == b_dt.kind, f"{a_dt.kind} != {b_dt.kind}"
    else:
        assert a_dt == b_dt, f"{a_dt} != {b_dt}"


def _assert_shape(a, b):
    # it is assumed `a` is a `dpnp.ndarray` and so it has shape attribute
    if hasattr(b, "shape"):
        assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    else:
        # numpy output is scalar, then dpnp is 0-D array
        assert a.shape == (), f"{a.shape} != ()"


def assert_dtype_allclose(
    dpnp_arr,
    numpy_arr,
    check_type=True,
    check_only_type_kind=False,
    factor=8,
    check_shape=True,
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
    The parameter `check_shape`, when True (default), asserts the shape of input arrays is the same.

    """

    if check_shape:
        _assert_shape(dpnp_arr, numpy_arr)

    is_inexact = lambda x: hasattr(x, "dtype") and dpnp.issubdtype(
        x.dtype, dpnp.inexact
    )

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
        assert_allclose(dpnp_arr, numpy_arr, atol=tol, rtol=tol, strict=False)
        if check_type:
            list_64bit_types = [numpy.float64, numpy.complex128]
            numpy_arr_dtype = numpy_arr.dtype
            dpnp_arr_dtype = dpnp_arr.dtype
            dpnp_arr_dev = dpnp_arr.sycl_device

            if check_only_type_kind:
                _assert_dtype(dpnp_arr_dtype, numpy_arr_dtype, True)
            else:
                is_np_arr_f2 = numpy_arr_dtype == numpy.float16

                if is_np_arr_f2:
                    if has_support_aspect16(dpnp_arr_dev):
                        _assert_dtype(dpnp_arr_dtype, numpy_arr_dtype)
                elif (
                    numpy_arr_dtype not in list_64bit_types
                    or has_support_aspect64(dpnp_arr_dev)
                ):
                    _assert_dtype(dpnp_arr_dtype, numpy_arr_dtype)
                else:
                    _assert_dtype(dpnp_arr_dtype, numpy_arr_dtype, True)
    else:
        assert_array_equal(dpnp_arr, numpy_arr, strict=False)
        if check_type and hasattr(numpy_arr, "dtype"):
            _assert_dtype(dpnp_arr.dtype, numpy_arr.dtype, check_only_type_kind)


def generate_random_numpy_array(
    shape,
    dtype=None,
    order="C",
    hermitian=False,
    seed_value=None,
    low=-10,
    high=10,
    probability=0.5,
):
    """
    Generate a random numpy array with the specified shape and dtype.

    If required, the array can be made Hermitian (for complex data types) or
    symmetric (for real data types).

    Parameters
    ----------
    shape : tuple
        Shape of the generated array.
    dtype : str or dtype, optional
        Desired data-type for the output array.
        If not specified, data type will be determined by numpy.

        Default : ``None``
    order : {"C", "F"}, optional
        Specify the memory layout of the output array.

        Default: ``"C"``.
    hermitian : bool, optional
        If True, generates a Hermitian (symmetric if `dtype` is real) matrix.

        Default : ``False``
    seed_value : int, optional
        The seed value to initialize the random number generator.

        Default : ``None``
    low : {int, float}, optional
        Lower boundary of the generated samples from a uniform distribution.

        Default : ``-10``.
    high : {int, float}, optional
        Upper boundary of the generated samples from a uniform distribution.

        Default : ``10``.
    probability : float, optional
        If dtype is bool, the probability of True. Ignored for other dtypes.

        Default : ``0.5``.
    Returns
    -------
    out : numpy.ndarray
        A random numpy array of the specified shape, dtype and memory layout.
        The array is Hermitian or symmetric if `hermitian` is True.

    Note:
    For arrays with more than 2 dimensions, the Hermitian or
    symmetric property is ensured for each 2D sub-array.

    """

    if seed_value is None:
        seed_value = 42
    numpy.random.seed(seed_value)

    if numpy.issubdtype(dtype, numpy.unsignedinteger):
        low = 0

    # dtype=int is needed for 0d arrays
    size = numpy.prod(shape, dtype=int)
    if dtype == dpnp.bool:
        a = numpy.random.choice(
            [False, True], size, p=[1 - probability, probability]
        )
    else:
        a = numpy.random.uniform(low, high, size).astype(dtype)

        if numpy.issubdtype(a.dtype, numpy.complexfloating):
            a += 1j * numpy.random.uniform(low, high, size)

    a = a.reshape(shape)
    if hermitian and a.size > 0:
        if a.ndim > 2:
            orig_shape = a.shape
            # get 3D array
            a = a.reshape(-1, orig_shape[-2], orig_shape[-1])
            for i in range(a.shape[0]):
                a[i] = numpy.conj(a[i].T) @ a[i]
            a = a.reshape(orig_shape)
        else:
            a = numpy.conj(a.T) @ a

    # a.reshape(shape) returns an array in C order by default
    if order != "C" and a.ndim > 1:
        a = numpy.array(a, order=order)
    return a


def factor_to_tol(dtype, factor):
    """
    Calculate the tolerance for comparing floating point and complex arrays.
    The tolerance is based on the maximum resolution of the input dtype multiplied by the factor.
    """

    tol = 0
    if numpy.issubdtype(dtype, numpy.inexact):
        tol = numpy.finfo(dtype).resolution

    return factor * tol


def get_abs_array(data, dtype=None):
    if numpy.issubdtype(dtype, numpy.unsignedinteger):
        data = numpy.abs(data)
    # it is better to use astype with the default casting=unsafe
    # otherwise, we need to skip test for cases where overflow occurs
    return numpy.array(data).astype(dtype)


def get_all_dtypes(
    no_bool=False,
    no_float16=True,
    no_complex=False,
    no_none=False,
    xfail_dtypes=None,
    exclude=None,
    no_unsigned=False,
    device=None,
):
    """
    Build a list of types supported by DPNP based on
    input flags and device capabilities.
    """

    dev = dpctl.select_default_device() if device is None else device

    # add boolean type
    dtypes = [dpnp.bool] if not no_bool else []

    # add integer types
    dtypes.extend(get_integer_dtypes(no_unsigned=no_unsigned))

    # add floating types
    dtypes.extend(get_float_dtypes(no_float16=no_float16, device=dev))

    # add complex types
    if not no_complex:
        dtypes.extend(get_complex_dtypes(device=dev))

    # add None value to validate a default dtype
    if not no_none:
        dtypes.append(None)

    def mark_xfail(dtype):
        if xfail_dtypes is not None and dtype in xfail_dtypes:
            return pytest.param(dtype, marks=pytest.mark.xfail)
        return dtype

    def not_excluded(dtype):
        if exclude is None:
            return True
        return dtype not in exclude

    dtypes = [mark_xfail(dtype) for dtype in dtypes if not_excluded(dtype)]
    return dtypes


def get_array(xp, a):
    """
    Cast input array `a` to a type supported by `xp` interface.

    Implicit conversion of either DPNP or DPCTL array to a NumPy array is not
    allowed. Input array has to be explicitly casted with `asnumpy` function.

    """

    if xp is numpy and dpnp.is_supported_array_type(a):
        return dpnp.asnumpy(a)
    return a


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


def get_integer_dtypes(all_int_types=False, no_unsigned=False):
    """
    Build a list of integer types supported by DPNP.
    """

    dtypes = [dpnp.int32, dpnp.int64]

    if config.all_int_types or all_int_types:
        dtypes += [dpnp.int8, dpnp.int16]
        if not no_unsigned:
            dtypes += [dpnp.uint8, dpnp.uint16, dpnp.uint32, dpnp.uint64]

    return dtypes


def get_integer_float_dtypes(
    all_int_types=False,
    no_unsigned=False,
    no_float16=True,
    device=None,
    xfail_dtypes=None,
    exclude=None,
):
    """
    Build a list of integer and float types supported by DPNP.
    """
    dtypes = get_integer_dtypes(
        all_int_types=all_int_types, no_unsigned=no_unsigned
    )
    dtypes += get_float_dtypes(no_float16=no_float16, device=device)

    def mark_xfail(dtype):
        if xfail_dtypes is not None and dtype in xfail_dtypes:
            return pytest.param(dtype, marks=pytest.mark.xfail)
        return dtype

    def not_excluded(dtype):
        if exclude is None:
            return True
        return dtype not in exclude

    dtypes = [mark_xfail(dtype) for dtype in dtypes if not_excluded(dtype)]
    return dtypes


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


def is_cpu_device(device=None):
    """
    Return True if a test is running on CPU device, False otherwise.
    """
    dev = dpctl.select_default_device() if device is None else device
    return dev.has_aspect_cpu


def is_cuda_device(device=None):
    """
    Return True if a test is running on CUDA device, False otherwise.
    """
    dev = dpctl.select_default_device() if device is None else device
    return dev.backend == dpctl.backend_type.cuda


def is_gpu_device(device=None):
    """
    Return True if a test is running on GPU device, False otherwise.
    """
    dev = dpctl.select_default_device() if device is None else device
    return dev.has_aspect_gpu


def is_intel_numpy():
    """
    Return True if Intel NumPy is used during testing.

    The check is based on MKL backend name stored in Build Dependencies, where
    in case of Intel Numpy there "mkl" is expected at the beginning of the name
    for both BLAS and LAPACK (the full name is "mkl-dynamic-ilp64-iomp").

    """

    build_deps = numpy.show_config(mode="dicts")["Build Dependencies"]
    blas = build_deps["blas"]
    lapack = build_deps["lapack"]

    if numpy_version() < "2.0.0":
        # numpy 1.26.4 has LAPACK name equals to 'dep140030038112336'
        return blas["name"].startswith("mkl")
    return all(dep["name"].startswith("mkl") for dep in [blas, lapack])


def is_win_platform():
    """
    Return True if a test is running on Windows OS, False otherwise.
    """
    return platform.startswith("win")


def numpy_version():
    return numpy.lib.NumpyVersion(numpy.__version__)


def requires_intel_mkl_version(version):
    """
    Check if Intel MKL is used and its version is greater than or
    equal to the specified one.

    The check is based on MKL backend name stored in Build Dependencies
    and only applies if Intel NumPy is detected.
    The version is extracted from the BLAS section of NumPy's build
    information and compared to the given version string.
    """
    if not is_intel_numpy():
        return False

    build_deps = numpy.show_config(mode="dicts")["Build Dependencies"]
    return build_deps["blas"]["version"] >= version
