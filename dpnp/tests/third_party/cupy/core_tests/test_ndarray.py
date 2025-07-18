import copy
import unittest

import dpctl
import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError

import dpnp as cupy
from dpnp.tests.third_party.cupy import testing


def get_array_module(*args):
    for arg in args:
        if isinstance(arg, cupy.ndarray):
            return cupy
    return numpy


def wrap_take(array, *args, **kwargs):
    if get_array_module(array) == numpy:
        kwargs["mode"] = "wrap"

    return array.take(*args, **kwargs)


class TestNdarrayInit(unittest.TestCase):

    def test_shape_none(self):
        with pytest.raises(TypeError):
            cupy.ndarray(None)

    def test_shape_int(self):
        a = cupy.ndarray(3)
        assert a.shape == (3,)

    def test_shape_not_integer(self):
        for xp in (numpy, cupy):
            with pytest.raises(TypeError):
                xp.ndarray(1.0)
            with pytest.raises(TypeError):
                xp.ndarray((1.0,))

    def test_shape_int_with_strides(self):
        dummy = cupy.ndarray(3)
        a = cupy.ndarray(3, strides=(0,), buffer=dummy.data)
        assert a.shape == (3,)
        assert a.strides == (0,)

    def test_memptr(self):
        a = cupy.arange(6).astype(numpy.float32).reshape((2, 3))
        memptr = a.data

        b = cupy.ndarray((2, 3), numpy.float32, memptr)
        testing.assert_array_equal(a, b)

        b += 1
        testing.assert_array_equal(a, b)

    @pytest.mark.skip(
        "dpctl-1765: might lead to race condition (no plan to support that)"
    )
    def test_memptr_with_strides(self):
        buf = cupy.ndarray(20, numpy.uint8)
        memptr = buf.data

        # self-overlapping strides
        a = cupy.ndarray((2, 3), numpy.float32, memptr, strides=(2, 1))
        assert a.strides == (2, 1)

        a[:] = 1
        a[0, 2] = 4
        assert float(a[1, 0]) == 4

    @pytest.mark.skip(
        "dpctl-1766: no exception raised by dpctl (no plan to support that)"
    )
    def test_strides_without_memptr(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.ndarray((2, 3), numpy.float32, strides=(20, 4))

    def test_strides_is_given_and_order_is_ignored(self):
        buf = cupy.ndarray(20, numpy.uint8)
        a = cupy.ndarray(
            (2, 3), numpy.float32, buf.data, strides=(2, 1), order="C"
        )
        assert a.strides == (2, 1)

    @testing.with_requires("numpy>=1.19")
    def test_strides_is_given_but_order_is_invalid(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.ndarray((2, 3), numpy.float32, strides=(2, 1), order="!")

    def test_order(self):
        shape = (2, 3, 4)
        a = cupy.ndarray(shape, order="F")
        a_cpu = numpy.ndarray(shape, order="F", dtype=a.dtype)
        assert all(
            i * a.itemsize == j for i, j in zip(a.strides, a_cpu.strides)
        )
        assert a.flags.f_contiguous
        assert not a.flags.c_contiguous

    def test_order_none(self):
        shape = (2, 3, 4)
        a = cupy.ndarray(shape, order=None)
        a_cpu = numpy.ndarray(shape, order=None, dtype=a.dtype)
        assert a.flags.c_contiguous == a_cpu.flags.c_contiguous
        assert a.flags.f_contiguous == a_cpu.flags.f_contiguous
        assert all(
            i * a.itemsize == j for i, j in zip(a.strides, a_cpu.strides)
        )

    def test_slots(self):
        # Test for #7883.
        a = cupy.ndarray((2, 3))
        with pytest.raises(AttributeError):
            a.custom_attr = 100

        class UserNdarray(cupy.ndarray):
            pass

        b = UserNdarray((2, 3))
        b.custom_attr = 100


@testing.parameterize(
    *testing.product(
        {
            "shape": [(), (1,), (2, 3), (1, 2, 3)],
            "order": ["C", "F"],
            "dtype": [
                numpy.uint8,  # itemsize=1
                numpy.uint16,  # itemsize=2
            ],
        }
    )
)
class TestNdarrayInitStrides(unittest.TestCase):

    # Check the strides given shape, itemsize and order.

    @testing.numpy_cupy_equal()
    def test_strides(self, xp):
        arr = xp.ndarray(self.shape, dtype=self.dtype, order=self.order)
        strides = arr.strides
        if xp is cupy:
            strides = tuple(i * arr.itemsize for i in strides)
        return (strides, arr.flags.c_contiguous, arr.flags.f_contiguous)


class TestNdarrayInitRaise(unittest.TestCase):

    def test_unsupported_type(self):
        arr = numpy.ndarray((2, 3), dtype=object)
        with pytest.raises(TypeError):
            cupy.array(arr)

    @pytest.mark.skip("no ndim limit")
    @testing.with_requires("numpy>=2.0")
    @testing.numpy_cupy_array_equal()
    def test_upper_limit_ndim(self, xp):
        shape = [1 for i in range(64)]
        return xp.zeros(shape, dtype=xp.int8)

    @pytest.mark.skip("no ndim limit")
    def test_excessive_ndim(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.ndarray(shape=[1 for i in range(65)], dtype=xp.int8)


@testing.parameterize(
    *testing.product(
        {
            "shape": [(), (0,), (1,), (0, 0, 2), (2, 3)],
        }
    )
)
@pytest.mark.skip("deepcopy() is not supported")
class TestNdarrayDeepCopy(unittest.TestCase):

    def _check_deepcopy(self, arr, arr2):
        assert arr.data is not arr2.data
        assert arr.shape == arr2.shape
        assert arr.size == arr2.size
        assert arr.dtype == arr2.dtype
        assert arr.strides == arr2.strides
        testing.assert_array_equal(arr, arr2)

    def test_deepcopy(self):
        arr = _core.ndarray(self.shape)
        arr2 = copy.deepcopy(arr)
        self._check_deepcopy(arr, arr2)

    @testing.multi_gpu(2)
    def test_deepcopy_multi_device(self):
        arr = _core.ndarray(self.shape)
        with cuda.Device(1):
            arr2 = copy.deepcopy(arr)
        self._check_deepcopy(arr, arr2)
        assert arr2.device == arr.device


_test_copy_multi_device_with_stream_src = r"""
extern "C" __global__
void wait_and_write(long long *x) {
  clock_t start = clock();
  clock_t now;
  for (;;) {
    now = clock();
    clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
    if (cycles >= 1000000000) {
      break;
    }
  }
  x[0] = 1;
  x[1] = now;  // in case the compiler optimizing away the entire loop
}
"""


class TestNdarrayCopy:

    @testing.multi_gpu(2)
    @testing.for_orders("CFA")
    def test_copy_multi_device_non_contiguous(self, order):
        arr = cupy.ndarray((20,))[::2]
        dev1 = dpctl.SyclDevice()
        arr2 = arr.copy(order, device=dev1)
        assert arr2.sycl_device == dev1
        testing.assert_array_equal(arr, arr2)

    @pytest.mark.skip("order='K' is supported in copy method")
    @testing.multi_gpu(2)
    def test_copy_multi_device_non_contiguous_K(self):
        arr = cupy.ndarray((20,))[::2]
        dev1 = dpctl.SyclDevice()
        with pytest.raises(NotImplementedError):
            arr.copy("K", device=dev1)

    # See cupy/cupy#5004
    @pytest.mark.skip("RawKernel() is not supported")
    @testing.multi_gpu(2)
    def test_copy_multi_device_with_stream(self):
        # Kernel that takes long enough then finally writes values.
        src = _test_copy_multi_device_with_stream_src
        if runtime.is_hip and driver.get_build_version() >= 5_00_00000:
            src = "#include <ctime>\n" + src
        kern = cupy.RawKernel(src, "wait_and_write")

        # Allocates a memory and launches the kernel on a device with its
        # stream.
        with cuda.Device(0):
            # Keep this stream alive over the D2D copy below for HIP
            with cuda.Stream() as s1:
                a = cupy.zeros((2,), dtype=numpy.uint64)
                kern((1,), (1,), a)

        # D2D copy to another device with another stream should get the
        # original values of the memory before the kernel on the first device
        # finally makes the write.
        with cuda.Device(1):
            with cuda.Stream():
                b = a.copy()
                testing.assert_array_equal(
                    b, numpy.array([0, 0], dtype=numpy.uint64)
                )


class TestNdarrayShape(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_shape_set(self, xp):
        arr = xp.ndarray((2, 3))
        arr.shape = (3, 2)
        return xp.array(arr.shape)

    @pytest.mark.skip(
        "dpctl-1699: shape setter does not work with negative shape "
        "(no plan to support that)"
    )
    @testing.numpy_cupy_array_equal()
    def test_shape_set_infer(self, xp):
        arr = xp.ndarray((2, 3))
        arr.shape = (3, -1)
        return xp.array(arr.shape)

    @testing.numpy_cupy_array_equal()
    def test_shape_set_int(self, xp):
        arr = xp.ndarray((2, 3))
        arr.shape = 6
        return xp.array(arr.shape)

    def test_shape_need_copy(self):
        # from cupy/cupy#5470
        for xp in (numpy, cupy):
            arr = xp.ndarray((2, 3), order="F")
            with pytest.raises(AttributeError) as e:
                arr.shape = (3, 2)
            assert "incompatible shape" in str(e.value).lower()


@pytest.mark.skip("CUDA interface is not supported")
class TestNdarrayCudaInterface(unittest.TestCase):

    def test_cuda_array_interface(self):
        arr = cupy.zeros(shape=(2, 3), dtype=cupy.float64)
        iface = arr.__cuda_array_interface__
        assert iface["version"] == 3
        assert set(iface.keys()) == set(
            [
                "shape",
                "typestr",
                "data",
                "version",
                "descr",
                "stream",
                "strides",
            ]
        )
        assert iface["shape"] == (2, 3)
        assert iface["typestr"] == "<f8"
        assert isinstance(iface["data"], tuple)
        assert len(iface["data"]) == 2
        assert iface["data"][0] == arr.data.ptr
        assert not iface["data"][1]
        assert iface["descr"] == [("", "<f8")]
        assert iface["strides"] is None
        assert iface["stream"] == stream_module.get_default_stream_ptr()

    def test_cuda_array_interface_view(self):
        arr = cupy.zeros(shape=(10, 20), dtype=cupy.float64)
        view = arr[::2, ::5]
        iface = view.__cuda_array_interface__
        assert iface["version"] == 3
        assert set(iface.keys()) == set(
            [
                "shape",
                "typestr",
                "data",
                "version",
                "descr",
                "stream",
                "strides",
            ]
        )
        assert iface["shape"] == (5, 4)
        assert iface["typestr"] == "<f8"
        assert isinstance(iface["data"], tuple)
        assert len(iface["data"]) == 2
        assert iface["data"][0] == arr.data.ptr
        assert not iface["data"][1]
        assert iface["strides"] == (320, 40)
        assert iface["descr"] == [("", "<f8")]
        assert iface["stream"] == stream_module.get_default_stream_ptr()

    def test_cuda_array_interface_zero_size(self):
        arr = cupy.zeros(shape=(10,), dtype=cupy.float64)
        view = arr[0:3:-1]
        iface = view.__cuda_array_interface__
        assert iface["version"] == 3
        assert set(iface.keys()) == set(
            [
                "shape",
                "typestr",
                "data",
                "version",
                "descr",
                "stream",
                "strides",
            ]
        )
        assert iface["shape"] == (0,)
        assert iface["typestr"] == "<f8"
        assert isinstance(iface["data"], tuple)
        assert len(iface["data"]) == 2
        assert iface["data"][0] == 0
        assert not iface["data"][1]
        assert iface["strides"] is None
        assert iface["descr"] == [("", "<f8")]
        assert iface["stream"] == stream_module.get_default_stream_ptr()


@testing.parameterize(
    *testing.product(
        {
            "stream": ("null", "new", "ptds"),
            "ver": (2, 3),
        }
    )
)
@pytest.mark.skip("CUDA interface is not supported")
class TestNdarrayCudaInterfaceStream(unittest.TestCase):
    def setUp(self):
        if self.stream == "null":
            self.stream = cuda.Stream.null
        elif self.stream == "new":
            self.stream = cuda.Stream()
        elif self.stream == "ptds":
            self.stream = cuda.Stream.ptds

        self.old_ver = _util.CUDA_ARRAY_INTERFACE_EXPORT_VERSION
        _util.CUDA_ARRAY_INTERFACE_EXPORT_VERSION = self.ver

    def tearDown(self):
        _util.CUDA_ARRAY_INTERFACE_EXPORT_VERSION = self.old_ver

    def test_cuda_array_interface_stream(self):
        # this tests exporting CAI with a given stream
        arr = cupy.zeros(shape=(10,), dtype=cupy.float64)
        stream = self.stream
        with stream:
            iface = arr.__cuda_array_interface__
        assert iface["version"] == self.ver
        attrs = ["shape", "typestr", "data", "version", "descr", "strides"]
        if self.ver == 3:
            attrs.append("stream")
        assert set(iface.keys()) == set(attrs)
        assert iface["shape"] == (10,)
        assert iface["typestr"] == "<f8"
        assert isinstance(iface["data"], tuple)
        assert len(iface["data"]) == 2
        assert iface["data"] == (arr.data.ptr, False)
        assert iface["descr"] == [("", "<f8")]
        assert iface["strides"] is None
        if self.ver == 3:
            if stream.ptr == 0:
                ptr = stream_module.get_default_stream_ptr()
                assert iface["stream"] == ptr
            else:
                assert iface["stream"] == stream.ptr


@pytest.mark.skip("CUDA interface is not supported")
class TestNdarrayCudaInterfaceNoneCUDA(unittest.TestCase):

    def setUp(self):
        self.arr = cupy.zeros(shape=(2, 3), dtype=cupy.float64)

    def test_cuda_array_interface_hasattr(self):
        assert not hasattr(self.arr, "__cuda_array_interface__")

    def test_cuda_array_interface_getattr(self):
        with pytest.raises(AttributeError) as e:
            getattr(self.arr, "__cuda_array_interface__")
        assert "HIP" in str(e.value)


@testing.parameterize(
    *testing.product(
        {
            "indices_shape": [(2,), (2, 3)],
            "axis": [None, 0, 1, 2, -1, -2],
        }
    )
)
class TestNdarrayTake(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.axis is None:
            m = a.size
        else:
            m = a.shape[self.axis]
        i = testing.shaped_arange(self.indices_shape, xp, numpy.int32) % m
        return wrap_take(a, i, self.axis)


@testing.parameterize(
    *testing.product(
        {
            "indices": [2, [0, 1], -1, [-1, -2]],
            "axis": [None, 0, 1, -1, -2],
        }
    )
)
class TestNdarrayTakeWithInt(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return wrap_take(a, self.indices, self.axis)


@testing.parameterize(
    *testing.product(
        {
            "indices": [2, [0, 1], -1, [-1, -2]],
            "axis": [None, 0, 1, -1, -2],
        }
    )
)
class TestNdarrayTakeWithIntWithOutParam(unittest.TestCase):

    shape = (3, 4, 5)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        r1 = wrap_take(a, self.indices, self.axis)
        r2 = xp.zeros_like(r1)
        wrap_take(a, self.indices, self.axis, out=r2)
        testing.assert_array_equal(r1, r2)
        return r2


@testing.parameterize(
    *testing.product(
        {
            "indices": [0, -1, [0], [0, -1]],
            "axis": [None, 0, -1],
        }
    )
)
class TestScalaNdarrayTakeWithIntWithOutParam(unittest.TestCase):

    shape = ()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_take(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        r1 = wrap_take(a, self.indices, self.axis)
        r2 = xp.zeros_like(r1)
        wrap_take(a, self.indices, self.axis, out=r2)
        testing.assert_array_equal(r1, r2)
        return r2


@testing.parameterize(
    {"shape": (3, 4, 5), "indices": (2,), "axis": 3},
    {"shape": (), "indices": (0,), "axis": 2},
)
class TestNdarrayTakeErrorAxisOverRun(unittest.TestCase):

    def test_axis_overrun1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp)
            with pytest.raises(AxisError):
                wrap_take(a, self.indices, axis=self.axis)

    def test_axis_overrun2(self):
        a = testing.shaped_arange(self.shape, cupy)
        with pytest.raises(AxisError):
            wrap_take(a, self.indices, axis=self.axis)


@testing.parameterize(
    {"shape": (3, 4, 5), "indices": (2, 3), "out_shape": (2, 4)},
    {"shape": (), "indices": (), "out_shape": (1,)},
)
class TestNdarrayTakeErrorShapeMismatch(unittest.TestCase):

    def test_shape_mismatch(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp)
            i = testing.shaped_arange(self.indices, xp, numpy.int32) % 3
            o = testing.shaped_arange(self.out_shape, xp)
            with pytest.raises(ValueError):
                wrap_take(a, i, out=o)


@testing.parameterize(
    {"shape": (3, 4, 5), "indices": (2, 3), "out_shape": (2, 3)},
    {"shape": (), "indices": (), "out_shape": ()},
)
class TestNdarrayTakeErrorTypeMismatch(unittest.TestCase):

    def test_output_type_mismatch(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, numpy.int32)
            i = testing.shaped_arange(self.indices, xp, numpy.int32) % 3
            o = testing.shaped_arange(self.out_shape, xp, numpy.float32)
            with pytest.raises(TypeError):
                wrap_take(a, i, out=o)


@testing.parameterize(
    {"shape": (0,), "indices": (0,), "axis": None},
    {"shape": (0,), "indices": (0, 1), "axis": None},
    {"shape": (3, 0), "indices": (2,), "axis": 0},
)
class TestZeroSizedNdarrayTake(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_output_type_mismatch(self, xp):
        a = testing.shaped_arange(self.shape, xp, numpy.int32)
        i = testing.shaped_arange(self.indices, xp, numpy.int32)
        return wrap_take(a, i, axis=self.axis)


@testing.parameterize(
    {"shape": (0,), "indices": (1,)},
    {"shape": (0,), "indices": (1, 1)},
)
class TestZeroSizedNdarrayTakeIndexError(unittest.TestCase):

    def test_output_type_mismatch(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, numpy.int32)
            i = testing.shaped_arange(self.indices, xp, numpy.int32)
            with pytest.raises(IndexError):
                wrap_take(a, i)


class TestSize(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test_size_without_axis(self, xp):
        x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
        return xp.size(x)

    @testing.numpy_cupy_equal()
    def test_size_with_axis(self, xp):
        x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
        return xp.size(x, 0)

    @testing.numpy_cupy_equal()
    def test_size_with_negative_axis(self, xp):
        x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
        return xp.size(x, -1)

    @testing.numpy_cupy_equal()
    def test_size_zero_dim_array(self, xp):
        x = testing.shaped_arange((), xp, numpy.int32)
        return xp.size(x)

    def test_size_axis_too_large(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
            with pytest.raises(IndexError):
                xp.size(x, 3)

    def test_size_axis_too_small(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((3, 4, 5), xp, numpy.int32)
            with pytest.raises(IndexError):
                xp.size(x, -4)

    def test_size_zero_dim_array_with_axis(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((), xp, numpy.int32)
            with pytest.raises(IndexError):
                xp.size(x, 0)


@pytest.mark.skip("python interface is not supported")
class TestPythonInterface(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_bytes_tobytes(self, xp, dtype):
        x = testing.shaped_arange((3, 4, 5), xp, dtype)
        return bytes(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_bytes_tobytes_empty(self, xp, dtype):
        x = xp.empty((0,), dtype)
        return bytes(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_bytes_tobytes_empty2(self, xp, dtype):
        x = xp.empty((3, 0, 4), dtype)
        return bytes(x)

    # The result of bytes(numpy.array(scalar)) is the same as bytes(scalar)
    # if scalar is of an integer dtype including bool_. It's spec is
    # bytes(int): bytes object of size given by the parameter initialized with
    # null bytes.
    @testing.for_float_dtypes()
    @testing.numpy_cupy_equal()
    def test_bytes_tobytes_scalar_array(self, xp, dtype):
        x = xp.array(3, dtype)
        return bytes(x)

    @testing.numpy_cupy_equal()
    def test_format(self, xp):
        x = xp.array(1.12345)
        return format(x, ".2f")


class TestNdarrayImplicitConversion(unittest.TestCase):

    def test_array(self):
        a = testing.shaped_arange((3, 4, 5), cupy, numpy.int64)
        with pytest.raises(TypeError):
            numpy.asarray(a)


class C(cupy.ndarray):

    def __new__(cls, *args, info=None, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, "info", None)


@pytest.mark.skip("SAT-7168: explicit constructor call is not supported")
class TestNdarraySubclass:

    def test_explicit_constructor_call(self):
        a = C([0, 1, 2, 3], info="information")
        assert type(a) is C
        assert issubclass(type(a), cupy.ndarray)
        assert a.info == "information"
