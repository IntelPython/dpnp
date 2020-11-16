import pytest
import dpnp
import numpy
import dpctl

def test_has_sycl_usm_array_interface():
    X = dpnp.arange(4)
    assert hasattr(X, '__sycl_usm_array_interface__')
    assert X.__sycl_usm_array_interface__['shape'] == X.shape
    assert numpy.dtype(X.__sycl_usm_array_interface__['typestr']) == X.dtype

def test_shared_memory():
    X = dpnp.arange(4, dtype=numpy.int32)
    ms = dpctl.memory.MemoryUSMShared(X)
    assert ms.nbytes == X.size * X.itemsize

    X_host = ms.copy_to_host()
    X_copied = X_host.view(X.dtype)
    assert all([ X_copied[i] == X[i] for i in range(len(X))])
    
    X_pattern = numpy.array([-7, 8, 18, -99], dtype=numpy.int32)
    ms.copy_from_host(X_pattern.view("|u1"))

    assert all([ X_pattern[i] == X[i] for i in range(len(X))])


class InvalidSyclUSMObject:
    def __init__(self, ary):
        self.ary = ary

    def __sycl_usm_array_interface__(self):
        ary_iface = self.ary.__array_interface__
        ary_iface['version'] = 1
        ary_iface['syclobj'] = dpctl.get_current_queue()
        return ary_iface

    
def test_non_shared_memory():
    md = dpctl.memory.MemoryUSMDevice(64)
    with pytest.raises(TypeError):
        # will fail because input is device memory
        X = dpnp.dparray.dparray((8,), memory=md)
        
    mh = dpctl.memory.MemoryUSMHost(64)
    with pytest.raises(TypeError):
        # will fail because input is host memory
        X = dpnp.dparray.dparray((8,), memory=mh)

    invalid_input = InvalidSyclUSMObject(
        numpy.arange(8, dtype='i8'))
    with pytest.raises(TypeError):
        # pointer has type unknown
        X = dpnp.dparray.dparray((8,), memory=invalid_input)
