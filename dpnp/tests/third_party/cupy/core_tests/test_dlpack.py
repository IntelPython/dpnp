import unittest

import dpctl
import dpctl.tensor._dlpack as dlp
import numpy
import pytest

import dpnp as cupy
from dpnp.tests.third_party.cupy import testing


# TODO: to roll back the changes once the issue with CUDA support is resolved for random
def _gen_array(dtype, alloc_q=None):
    if cupy.issubdtype(dtype, numpy.unsignedinteger):
        array = numpy.random.randint(0, 10, size=(2, 3))
    elif cupy.issubdtype(dtype, cupy.integer):
        array = numpy.random.randint(-10, 10, size=(2, 3))
    elif cupy.issubdtype(dtype, cupy.floating):
        array = numpy.random.rand(2, 3)
    elif cupy.issubdtype(dtype, cupy.complexfloating):
        array = numpy.random.random((2, 3))
    elif dtype == cupy.bool_:
        array = numpy.random.randint(0, 2, size=(2, 3))
    else:
        assert False, f"unrecognized dtype: {dtype}"
    return cupy.asarray(array, sycl_queue=alloc_q).astype(dtype)


class TestDLPackConversion(unittest.TestCase):
    @testing.for_all_dtypes(no_bool=False)
    def test_conversion(self, dtype):
        orig_array = _gen_array(dtype)
        tensor = orig_array.__dlpack__()
        out_array = dlp.from_dlpack_capsule(tensor)
        testing.assert_array_equal(orig_array, out_array)
        assert orig_array.get_array()._pointer == out_array._pointer


@testing.parameterize(*testing.product({"memory": ("device", "managed")}))
class TestNewDLPackConversion(unittest.TestCase):
    def _get_stream(self, stream_name):
        if stream_name == "null":
            return dpctl.SyclQueue()
        return dpctl.SyclQueue()

    @testing.for_all_dtypes(no_bool=False)
    def test_conversion(self, dtype):
        orig_array = _gen_array(dtype)
        out_array = cupy.from_dlpack(orig_array)
        testing.assert_array_equal(orig_array, out_array)
        assert orig_array.get_array()._pointer == out_array.get_array()._pointer

    def test_stream(self):
        allowed_streams = ["null", True]

        # stream order is automatically established via DLPack protocol
        for src_s in [self._get_stream(s) for s in allowed_streams]:
            for dst_s in [self._get_stream(s) for s in allowed_streams]:
                orig_array = _gen_array(cupy.float32, alloc_q=src_s)
                dltensor = orig_array.__dlpack__(stream=orig_array)

                out_array = dlp.from_dlpack_capsule(dltensor)
                out_array = cupy.from_dlpack(out_array, device=dst_s)
                testing.assert_array_equal(orig_array, out_array)
                assert (
                    orig_array.get_array()._pointer
                    == out_array.get_array()._pointer
                )


class TestDLTensorMemory(unittest.TestCase):
    # def setUp(self):
    #     self.old_pool = cupy.get_default_memory_pool()
    #     self.pool = cupy.cuda.MemoryPool()
    #     cupy.cuda.set_allocator(self.pool.malloc)

    # def tearDown(self):
    #     self.pool.free_all_blocks()
    #     cupy.cuda.set_allocator(self.old_pool.malloc)

    def test_deleter(self):
        # memory is freed when tensor is deleted, as it's not consumed
        array = cupy.empty(10)
        tensor = array.__dlpack__()
        # str(tensor): <capsule object "dltensor" at 0x7f7c4c835330>
        assert '"dltensor"' in str(tensor)
        # assert self.pool.n_free_blocks() == 0
        # del array
        # assert self.pool.n_free_blocks() == 0
        # del tensor
        # assert self.pool.n_free_blocks() == 1

    def test_deleter2(self):
        # memory is freed when array2 is deleted, as tensor is consumed
        array = cupy.empty(10)
        tensor = array.__dlpack__()
        assert '"dltensor"' in str(tensor)
        array2 = dlp.from_dlpack_capsule(tensor)
        assert '"used_dltensor"' in str(tensor)
        # assert self.pool.n_free_blocks() == 0
        # del array
        # assert self.pool.n_free_blocks() == 0
        # del array2
        # assert self.pool.n_free_blocks() == 1
        # del tensor
        # assert self.pool.n_free_blocks() == 1

    def test_multiple_consumption_error(self):
        # Prevent segfault, see #3611
        array = cupy.empty(10)
        tensor = array.__dlpack__()
        array2 = dlp.from_dlpack_capsule(tensor)
        with pytest.raises(ValueError) as e:
            array3 = dlp.from_dlpack_capsule(tensor)
        assert "consumed multiple times" in str(e.value)
