import dpctl
import dpctl.memory as dpmem
import dpnp
import numpy as np
import pytest
import numba_dppy as dppy
import dpctl.tensor.numpy_usm_shared as usmarray


class DuckUSMArray:
    def __init__(self, shape, dtype="d", host_buffer=None):
        nelems = np.prod(shape)
        bytes = nelems * np.dtype(dtype).itemsize
        shmem = dpmem.MemoryUSMShared(bytes)
        if isinstance(host_buffer, np.ndarray):
            shmem.copy_from_host(host_buffer.view(dtype="|u1"))
        self.arr = np.ndarray(shape, dtype=dtype, buffer=shmem)

    def __getitem__(self, indx):
        return self.arr[indx]

    def __setitem__(self, indx, val):
        self.arr.__setitem__(indx, val)

    @property
    def __sycl_usm_array_interface__(self):
        iface = self.arr.__array_interface__
        b = self.arr.base
        iface["syclobj"] = b.__sycl_usm_array_interface__["syclobj"]
        iface["version"] = 1
        return iface


def test_dpnp_interaction_with_dpctl_memory():
    """Tests if dpnp supports zero-copy data exchange with another Python
    object that defines `__sycl_usm_array_interface__`
    """
    hb = np.arange(0, 100, dtype=np.int64)
    da = DuckUSMArray(hb.shape, dtype=hb.dtype, host_buffer=hb)

    Y = dpnp.asarray(da)
    # dpnp array must infer dimensions/dtype from input object
    assert Y.dtype == hb.dtype
    assert Y.shape == hb.shape

    Y[0] = 10
    assert da[0] == 10  # check zero copy


def test_dppy_array_pass():
    """Tests if dppy supports passing an array-like object DuckArray that defines `__sycl_usm_array_interface__`
    to a dppy.kernel
    """

    @dppy.kernel
    def dppy_f(array_like_obj):
        i = dppy.get_global_id(0)
        array_like_obj[i] = 10

    global_size = 100
    hb = np.arange(0, global_size, dtype="i4")
    da = DuckUSMArray(hb.shape, dtype=hb.dtype, host_buffer=hb)

    if dpctl.has_gpu_queues(dpctl.backend_type.level_zero):
        print("\nScheduling on OpenCL GPU\n")
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppy_f[global_size, dppy.DEFAULT_LOCAL_SIZE](da)
    else:
        print("\nSkip scheduling on OpenCL GPU\n")

    assert da[0] == 10


def test_dpctl_dparray_has_iface():
    """Tests if dpctl.dptensor.numpy_usm_shared defines '__sycl_usm_array_interface__'"""
    X = usmarray.ones(10)
    assert type(getattr(X, "__sycl_usm_array_interface__", None) is dict)


def test_dpnp_array_has_iface():
    """Tests if dpnp.ndarray defines '__sycl_usm_array_interface__'"""
    X = dpnp.array([1])
    assert type(getattr(X, "__sycl_usm_array_interface__", None) is dict)
