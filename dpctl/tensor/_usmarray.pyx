# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

import dpctl
import dpctl.memory as dpmem
import numpy as np

from dpctl._backend cimport DPCTLSyclUSMRef
from dpctl._sycl_device_factory cimport _cached_default_device

from ._data_types import bool as dpt_bool
from ._device import Device
from ._print import usm_ndarray_repr, usm_ndarray_str

cimport dpctl as c_dpctl
cimport dpctl.memory as c_dpmem
from cpython.mem cimport PyMem_Free
from cpython.tuple cimport PyTuple_New, PyTuple_SetItem

# TODO: replace with
# cimport dpctl.tensor._dlpack as c_dlpack
# when dpctl.tensor is removed from dpctl
cimport ._dlpack as c_dlpack

from enum import IntEnum

# TODO: migrate dpctl.tensor._tensor_impl into dpnp
from dpctl.tensor._tensor_impl import default_device_fp_type

# TODO: replace with
# import dpctl.tensor._flags as _flags
# when dpctl.tensor is removed from dpctl
from . import _flags
from ._dlpack import get_build_dlpack_version

include "_stride_utils.pxi"
include "_types.pxi"
include "_slicing.pxi"


class DLDeviceType(IntEnum):
    """
    An :class:`enum.IntEnum` for the types of DLDevices supported by the DLPack
    protocol.

        ``kDLCPU``:
            CPU (host) device
        ``kDLCUDA``:
            CUDA GPU device
        ``kDLCUDAHost``:
            Pinned CUDA CPU memory by cudaMallocHost
        ``kDLOpenCL``:
            OpenCL device
        ``kDLVulkan``:
            Vulkan buffer
        ``kDLMetal``:
            Metal for Apple GPU
        ``kDLVPI``:
            Verilog simulator buffer
        ``kDLROCM``:
            ROCm GPU device
        ``kDLROCMHost``:
            Pinned ROCm CPU memory allocated by hipMallocHost
        ``kDLExtDev``:
            Reserved extension device type used to test new devices
        ``kDLCUDAManaged``:
            CUDA managed/unified memory allocated by cudaMallocManaged
        ``kDLOneAPI``:
            Unified shared memory allocated on a oneAPI non-partitioned device
        ``kDLWebGPU``:
            Device support for WebGPU standard
        ``kDLHexagon``:
            Qualcomm Hexagon DSP
        ``kDLMAIA``:
            Microsoft MAIA device
        ``kDLTrn``:
            AWS Trainium device
    """
    kDLCPU = c_dlpack.device_CPU
    kDLCUDA = c_dlpack.device_CUDA
    kDLCUDAHost = c_dlpack.device_CUDAHost
    kDLCUDAManaged = c_dlpack.device_CUDAManaged
    kDLROCM = c_dlpack.device_DLROCM
    kDLROCMHost = c_dlpack.device_ROCMHost
    kDLOpenCL = c_dlpack.device_OpenCL
    kDLVulkan = c_dlpack.device_Vulkan
    kDLMetal = c_dlpack.device_Metal
    kDLVPI = c_dlpack.device_VPI
    kDLOneAPI = c_dlpack.device_OneAPI
    kDLWebGPU = c_dlpack.device_WebGPU
    kDLHexagon = c_dlpack.device_Hexagon
    kDLMAIA = c_dlpack.device_MAIA
    kDLTrn = c_dlpack.device_Trn


cdef class InternalUSMArrayError(Exception):
    """
    An InternalUSMArrayError exception is raised when internal
    inconsistency has been detected in :class:`.usm_ndarray`.
    """
    pass


cdef object _as_zero_dim_ndarray(object usm_ary):
    "Convert size-1 array to NumPy 0d array"
    mem_view = dpmem.as_usm_memory(usm_ary)
    usm_ary.sycl_queue.wait()
    host_buf = mem_view.copy_to_host()
    view = host_buf.view(usm_ary.dtype)
    view.shape = tuple()
    return view


cdef inline void _check_0d_scalar_conversion(object usm_ary) except *:
    "Raise TypeError if array cannot be converted to a Python scalar"
    if (usm_ary.ndim != 0):
        raise TypeError(
            "only 0-dimensional arrays can be converted to Python scalars"
        )


cdef int _copy_writable(int lhs_flags, int rhs_flags):
    "Copy the WRITABLE flag to lhs_flags from rhs_flags"
    return (lhs_flags & ~USM_ARRAY_WRITABLE) | (rhs_flags & USM_ARRAY_WRITABLE)


cdef bint _is_host_cpu(object dl_device):
    "Check if dl_device denotes (kDLCPU, 0)"
    cdef object dl_type
    cdef object dl_id
    cdef Py_ssize_t n_elems = -1

    try:
        n_elems = len(dl_device)
    except TypeError:
        pass

    if n_elems != 2:
        return False

    dl_type = dl_device[0]
    dl_id = dl_device[1]
    if isinstance(dl_type, str):
        return (dl_type == "kDLCPU" and dl_id == 0)

    return (dl_type == DLDeviceType.kDLCPU) and (dl_id == 0)


cdef void _validate_and_use_stream(
    object stream, c_dpctl.SyclQueue self_queue
) except *:
    if (stream is None or stream == self_queue):
        pass
    else:
        if not isinstance(stream, dpctl.SyclQueue):
            raise TypeError(
                "stream argument type was expected to be dpctl.SyclQueue,"
                f" got {type(stream)} instead"
            )
        ev = self_queue.submit_barrier()
        stream.submit_barrier(dependent_events=[ev])

cdef class usm_ndarray:
    """ usm_ndarray(shape, dtype=None, strides=None, buffer="device", \
           offset=0, order="C", buffer_ctor_kwargs=dict(), \
           array_namespace=None)

    An array object represents a multidimensional tensor of numeric
    elements stored in a USM allocation on a SYCL device.

    Arg:
        shape (int, tuple):
            Shape of the array to be created.
        dtype (str, dtype):
            Array data type, i.e. the type of array elements.
            If ``dtype`` has the value ``None``, it is determined by default
            floating point type supported by target device.
            The supported types are

                ``bool``:
                    boolean type
                ``int8``, ``int16``, ``int32``, ``int64``:
                    signed integer types
                ``uint8``, ``uint16``, ``uint32``, ``uint64``:
                    unsigned integer types
                ``float16``:
                    half-precision floating type,
                    supported if target device's property
                    ``has_aspect_fp16`` is ``True``
                ``float32``, ``complex64``:
                    single-precision real and complex floating types
                ``float64``, ``complex128``:
                    double-precision real and complex floating
                    types, supported if target device's property
                    ``has_aspect_fp64`` is ``True``.

            Default: ``None``.
        strides (tuple, optional):
            Strides of the array to be created in elements.
            If ``strides`` has the value ``None``, it is determined by the
            ``shape`` of the array and the requested ``order``.
            Default: ``None``.
        buffer (str, object, optional):
            A string corresponding to the type of USM allocation to make,
            or a Python object representing a USM memory allocation, i.e.
            :class:`dpctl.memory.MemoryUSMDevice`,
            :class:`dpctl.memory.MemoryUSMShared`, or
            :class:`dpctl.memory.MemoryUSMHost`. Recognized strings are
            ``"device"``, ``"shared"``, or ``"host"``. Additional arguments to
            the USM memory allocators can be passed in a dictionary specified
            via ``buffer_ctor_kwrds`` keyword parameter.
            Default: ``"device"``.
        offset (int, optional):
            Offset of the array element with all zero indexes relative to the
            start of the provided `buffer` in elements. The argument is ignored
            if the ``buffer`` value is a string and the memory is allocated by
            the constructor. Default: ``0``.
        order ({"C", "F"}, optional):
            The memory layout of the array when constructing using a new
            allocation. Value ``"C"`` corresponds to C-contiguous, or row-major
            memory layout, while value ``"F"`` corresponds to F-contiguous, or
            column-major layout. Default: ``"C"``.
        buffer_ctor_kwargs (dict, optional):
            Dictionary with keyword parameters to use when creating a new USM
            memory allocation. See :class:`dpctl.memory.MemoryUSMShared` for
            supported keyword arguments.
        array_namespace (module, optional):
            Array namespace module associated with this array.
            Default: ``None``.

    ``buffer`` can be ``"shared"``, ``"host"``, ``"device"`` to allocate
    new device memory by calling respective constructor with
    the specified ``buffer_ctor_kwrds``; ``buffer`` can be an
    instance of :class:`dpctl.memory.MemoryUSMShared`,
    :class:`dpctl.memory.MemoryUSMDevice`, or
    :class:`dpctl.memory.MemoryUSMHost`; ``buffer`` can also be
    another :class:`dpctl.tensor.usm_ndarray` instance, in which case its
    underlying ``MemoryUSM*`` buffer is used.
    """

    cdef void _reset(usm_ndarray self):
        """
        Initializes member fields
        """
        self.base_ = None
        self.array_namespace_ = None
        self.nd_ = -1
        self.data_ = <char *>0
        self.shape_ = <Py_ssize_t *>0
        self.strides_ = <Py_ssize_t *>0
        self.flags_ = 0

    cdef void _cleanup(usm_ndarray self):
        if (self.shape_):
            PyMem_Free(self.shape_)
        if (self.strides_):
            PyMem_Free(self.strides_)
        self._reset()

    def __cinit__(self, shape, dtype=None, strides=None, buffer="device",
                  Py_ssize_t offset=0, order="C",
                  buffer_ctor_kwargs=dict(),
                  array_namespace=None):
        """
        strides and offset must be given in units of array elements.
        buffer can be strings ('device'|'shared'|'host' to allocate new memory)
        or ``dpctl.memory.MemoryUSM*`` buffers, or ``usm_ndarray`` instances.
        """
        cdef int nd = 0
        cdef int typenum = 0
        cdef int itemsize = 0
        cdef int err = 0
        cdef int contig_flag = 0
        cdef int writable_flag = USM_ARRAY_WRITABLE
        cdef Py_ssize_t *shape_ptr = NULL
        cdef Py_ssize_t ary_nelems = 0
        cdef Py_ssize_t ary_nbytes = 0
        cdef Py_ssize_t *strides_ptr = NULL
        cdef Py_ssize_t _offset = offset
        cdef Py_ssize_t ary_min_displacement = 0
        cdef Py_ssize_t ary_max_displacement = 0
        cdef bint is_fp64 = False
        cdef bint is_fp16 = False

        self._reset()
        if not isinstance(shape, (list, tuple)):
            if hasattr(shape, "tolist"):
                fn = getattr(shape, "tolist")
                if callable(fn):
                    shape = shape.tolist()
            if not isinstance(shape, (list, tuple)):
                try:
                    <Py_ssize_t> shape
                    shape = [shape, ]
                except Exception as e:
                    raise TypeError(
                        "Argument shape must a non-negative integer, "
                        "or a list/tuple of such integers."
                    ) from e
        nd = len(shape)
        if dtype is None:
            if isinstance(buffer, (dpmem._memory._Memory, usm_ndarray)):
                q = buffer.sycl_queue
            else:
                q = buffer_ctor_kwargs.get("queue")
            if q is not None:
                dtype = default_device_fp_type(q)
            else:
                dev = _cached_default_device()
                dtype = "f8" if dev.has_aspect_fp64 else "f4"
        typenum = dtype_to_typenum(dtype)
        if (typenum < 0):
            if typenum == -2:
                raise ValueError(
                    "Data type '" + str(dtype) +
                    "' can only have native byteorder."
                )
            elif typenum == -1:
                raise ValueError(
                    "Data type '" + str(dtype) + "' is not understood."
                )
            raise TypeError(
                f"Expected string or a dtype object, got {type(dtype)}"
            )
        itemsize = type_bytesize(typenum)
        if (itemsize < 1):
            raise TypeError(
                "dtype=" + np.dtype(dtype).name + " is not supported."
            )
        # allocate host C-arrays for shape, strides
        err = _from_input_shape_strides(
            nd, shape, strides, itemsize, <char> ord(order),
            &shape_ptr, &strides_ptr, &ary_nelems,
            &ary_min_displacement, &ary_max_displacement, &contig_flag
        )
        if (err):
            self._cleanup()
            if err == ERROR_MALLOC:
                raise MemoryError("Memory allocation for shape/strides "
                                  "array failed.")
            elif err == ERROR_INCORRECT_ORDER:
                raise ValueError(
                    "Unsupported order='{}' given. "
                    "Supported values are 'C' or 'F'.".format(order))
            elif err == ERROR_UNEXPECTED_STRIDES:
                raise ValueError(
                    "strides={} is not understood".format(strides))
            else:
                raise InternalUSMArrayError(
                    " .. while processing shape and strides.")
        ary_nbytes = (ary_max_displacement -
                      ary_min_displacement + 1) * itemsize
        if isinstance(buffer, dpmem._memory._Memory):
            _buffer = buffer
        elif isinstance(buffer, (str, bytes)):
            if isinstance(buffer, bytes):
                buffer = buffer.decode("UTF-8")
            _offset = -ary_min_displacement
            if (buffer == "shared"):
                _buffer = dpmem.MemoryUSMShared(ary_nbytes,
                                                **buffer_ctor_kwargs)
            elif (buffer == "device"):
                _buffer = dpmem.MemoryUSMDevice(ary_nbytes,
                                                **buffer_ctor_kwargs)
            elif (buffer == "host"):
                _buffer = dpmem.MemoryUSMHost(ary_nbytes,
                                              **buffer_ctor_kwargs)
            else:
                self._cleanup()
                raise ValueError(
                    "buffer='{}' is not understood. "
                    "Recognized values are 'device', 'shared',  'host', "
                    "an instance of `MemoryUSM*` object, or a usm_ndarray"
                    "".format(buffer)
                )
        elif isinstance(buffer, usm_ndarray):
            if not buffer.flags.writable:
                writable_flag = 0
            _buffer = buffer.usm_data
        else:
            self._cleanup()
            raise ValueError("buffer='{}' was not understood.".format(buffer))
        if (shape_to_elem_count(nd, shape_ptr) > 0 and
            (_offset + ary_min_displacement < 0 or
             (_offset + ary_max_displacement + 1) * itemsize > _buffer.nbytes)):
            self._cleanup()
            raise ValueError(("buffer='{}' can not accommodate "
                              "the requested array.").format(buffer))
        is_fp64 = (typenum == UAR_DOUBLE or typenum == UAR_CDOUBLE)
        is_fp16 = (typenum == UAR_HALF)
        if (is_fp64 or is_fp16):
            if (
                (is_fp64 and not _buffer.sycl_device.has_aspect_fp64) or
                (is_fp16 and not _buffer.sycl_device.has_aspect_fp16)
            ):
                raise ValueError(
                    f"Device {_buffer.sycl_device.name} does"
                    f" not support {dtype} natively."
                )
        self.base_ = _buffer
        self.data_ = (<char *> (<size_t> _buffer._pointer)) + itemsize * _offset
        self.shape_ = shape_ptr
        self.strides_ = strides_ptr
        self.typenum_ = typenum
        self.flags_ = (contig_flag | writable_flag)
        self.nd_ = nd
        self.array_namespace_ = array_namespace

    def __dealloc__(self):
        self._cleanup()

    @property
    def _pointer(self):
        """
        Returns USM pointer to the start of array (element with zero
        multi-index) encoded as integer.
        """
        return <size_t> self.get_data()

    cdef Py_ssize_t get_offset(self) except *:
        cdef char *mem_ptr = NULL
        cdef char *ary_ptr = self.get_data()
        mem_ptr = <char *>(<size_t> self.base_._pointer)
        byte_offset = ary_ptr - mem_ptr
        item_size = self.get_itemsize()
        if (byte_offset % item_size):
            raise InternalUSMArrayError(
                "byte_offset is not a multiple of item_size.")
        return byte_offset // item_size

    @property
    def _element_offset(self):
        """Returns the offset of the zero-index element of the array, in
        elements, relative to the start of memory allocation"""
        return self.get_offset()

    @property
    def _byte_bounds(self):
        """Returns a 2-tuple with pointers to the end-points of the array

        :Example:

            .. code-block:: python

                from dpctl import tensor

                x = tensor.ones((3, 10, 7))
                y = tensor.flip(x[:, 1::2], axis=1)

                beg_p, end_p = y._byte_bounds
                # Bytes taken to store this array
                bytes_extent = end_p - beg_p

                # C-contiguous copy is more compact
                yc = tensor.copy(y, order="C")
                beg_pc, end_pc = yc._byte_bounds
                assert bytes_extent < end_pc - beg_pc
        """
        cdef Py_ssize_t min_disp = 0
        cdef Py_ssize_t max_disp = 0
        cdef Py_ssize_t step_ = 0
        cdef Py_ssize_t dim_ = 0
        cdef int it = 0
        cdef Py_ssize_t _itemsize = self.get_itemsize()

        if (
            (self.flags_ & USM_ARRAY_C_CONTIGUOUS)
            or (self.flags_ & USM_ARRAY_F_CONTIGUOUS)
        ):
            return (
                self._pointer,
                self._pointer + shape_to_elem_count(
                    self.nd_, self.shape_
                ) * _itemsize
            )

        for it in range(self.nd_):
            dim_ = self.shape[it]
            if dim_ > 0:
                step_ = self.strides[it]
                if step_ > 0:
                    max_disp += step_ * (dim_ - 1)
                else:
                    min_disp += step_ * (dim_ - 1)

        return (
            self._pointer + min_disp * _itemsize,
            self._pointer + (max_disp + 1) * _itemsize
        )

    cdef char* get_data(self):
        """Returns the USM pointer for this array."""
        return self.data_

    cdef int get_ndim(self):
        """
        Returns the number of indices needed to address
        an element of this array.
        """
        return self.nd_

    cdef Py_ssize_t* get_shape(self):
        """
        Returns pointer to shape C-array for this array.

        C-array has at least ``ndim`` non-negative elements,
        which determine the range of permissible indices
        addressing individual elements of this array.
        """
        return self.shape_

    cdef Py_ssize_t* get_strides(self):
        """
        Returns pointer to strides C-array for this array.

        The pointer can be NULL (contiguous array), or the
        array size is at least ``ndim`` elements
        """
        return self.strides_

    cdef int get_typenum(self):
        """Returns typenum corresponding to values of this array"""
        return self.typenum_

    cdef int get_itemsize(self):
        """
        Returns itemsize of this arrays in bytes
        """
        return type_bytesize(self.typenum_)

    cdef int get_flags(self):
        """Returns flags of this array"""
        return self.flags_

    cdef object get_base(self):
        """Returns the object owning the USM data addressed by this array"""
        return self.base_

    cdef c_dpctl.SyclQueue get_sycl_queue(self):
        cdef c_dpmem._Memory mem
        if not isinstance(self.base_, dpctl.memory._Memory):
            raise InternalUSMArrayError(
                "This array has unexpected memory owner"
            )
        mem = <c_dpmem._Memory> self.base_
        return mem.queue

    cdef c_dpctl.DPCTLSyclQueueRef get_queue_ref(self) except *:
        """
        Returns a copy of DPCTLSyclQueueRef associated with array
        """
        cdef c_dpctl.SyclQueue q = self.get_sycl_queue()
        cdef c_dpctl.DPCTLSyclQueueRef QRef = q.get_queue_ref()
        cdef c_dpctl.DPCTLSyclQueueRef QRefCopy = NULL
        if QRef is not NULL:
            QRefCopy = c_dpctl.DPCTLQueue_Copy(QRef)
            return QRefCopy
        else:
            raise InternalUSMArrayError(
                "Memory owner of this array is corrupted"
            )

    @property
    def __sycl_usm_array_interface__(self):
        """
        Gives ``__sycl_usm_array_interface__`` dictionary describing
        the array.
        """
        cdef Py_ssize_t byte_offset = -1
        cdef int item_size = -1
        cdef Py_ssize_t elem_offset = -1
        cdef char *mem_ptr = NULL
        cdef char *ary_ptr = NULL
        if (not isinstance(self.base_, dpmem._memory._Memory)):
            raise InternalUSMArrayError(
                "Invalid instance of usm_ndarray encountered. "
                "Private field base_ has an unexpected type {}.".format(
                    type(self.base_)
                )
            )
        ary_iface = self.base_.__sycl_usm_array_interface__
        mem_ptr = <char *>(<size_t> ary_iface["data"][0])
        ary_ptr = <char *>(<size_t> self.data_)
        ro_flag = False if (self.flags_ & USM_ARRAY_WRITABLE) else True
        ary_iface["data"] = (<size_t> mem_ptr, ro_flag)
        ary_iface["shape"] = self.shape
        if (self.strides_):
            ary_iface["strides"] = _make_int_tuple(self.nd_, self.strides_)
        else:
            if (self.flags_ & USM_ARRAY_C_CONTIGUOUS):
                ary_iface["strides"] = None
            elif (self.flags_ & USM_ARRAY_F_CONTIGUOUS):
                ary_iface["strides"] = _f_contig_strides(self.nd_, self.shape_)
            else:
                raise InternalUSMArrayError(
                    "USM Array is not contiguous and has empty strides"
                )
        ary_iface["typestr"] = _make_typestr(self.typenum_)
        byte_offset = ary_ptr - mem_ptr
        item_size = self.get_itemsize()
        if (byte_offset % item_size):
            raise InternalUSMArrayError(
                "byte_offset is not a multiple of item_size.")
        elem_offset = byte_offset // item_size
        ary_iface["offset"] = elem_offset
        # must wait for content of the memory to finalize
        self.sycl_queue.wait()
        return ary_iface

    @property
    def ndim(self):
        """
        Gives the number of indices needed to address elements of this array.
        """
        return self.nd_

    @property
    def usm_data(self):
        """
        Gives USM memory object underlying :class:`.usm_ndarray` instance.
        """
        return self.get_base()

    @property
    def shape(self):
        """
        Elements of the shape tuple give the lengths of the
        respective array dimensions.

        Setting shape is allowed only when reshaping to the requested
        dimensions can be returned as view, otherwise :exc:`AttributeError`
        is raised. Use :func:`dpctl.tensor.reshape` to reshape the array
        in all cases.

        :Example:

            .. code-block:: python

                from dpctl import tensor

                x = tensor.arange(899)
                x.shape = (29, 31)
        """
        if self.nd_ > 0:
            return _make_int_tuple(self.nd_, self.shape_)
        else:
            return tuple()

    @shape.setter
    def shape(self, new_shape):
        """
        Modifies usm_ndarray instance in-place by changing its metadata
        about the shape and the strides of the array, or raises
        `AttributeError` exception if in-place change is not possible.

        Args:
            new_shape: (tuple, int)
                New shape. Only non-negative values are supported.
                The new shape may not lead to the change in the
                number of elements in the array.

        Whether the array can be reshape in-place depends on its
        strides. Use :func:`dpctl.tensor.reshape` function which
        always succeeds to reshape the array by performing a copy
        if necessary.
        """
        cdef int new_nd = -1
        cdef Py_ssize_t nelems = -1
        cdef int err = 0
        cdef Py_ssize_t min_disp = 0
        cdef Py_ssize_t max_disp = 0
        cdef int contig_flag = 0
        cdef Py_ssize_t *shape_ptr = NULL
        cdef Py_ssize_t *strides_ptr = NULL
        cdef Py_ssize_t size = -1
        import operator

        from ._reshape import reshaped_strides

        try:
            new_nd = len(new_shape)
        except TypeError:
            new_nd = 1
            new_shape = (new_shape,)
        try:
            new_shape = tuple(operator.index(dim) for dim in new_shape)
        except TypeError:
            raise TypeError(
                "Target shape must be a finite iterable of integers"
            )
        size = shape_to_elem_count(self.nd_, self.shape_)
        if not np.prod(new_shape) == size:
            raise TypeError(
                f"Can not reshape array of size {self.size} into {new_shape}"
            )
        if size > 0:
            new_strides = reshaped_strides(
               self.shape,
               self.strides,
               new_shape
            )
        else:
            new_strides = (1,) * len(new_shape)
        if new_strides is None:
            raise AttributeError(
                "Incompatible shape for in-place modification. "
                "Use `reshape()` to make a copy with the desired shape."
            )
        err = _from_input_shape_strides(
            new_nd, new_shape, new_strides,
            self.get_itemsize(),
            b"C",
            &shape_ptr, &strides_ptr,
            &nelems, &min_disp, &max_disp, &contig_flag
        )
        if (err == 0):
            if (self.shape_):
                PyMem_Free(self.shape_)
            if (self.strides_):
                PyMem_Free(self.strides_)
            self.flags_ = (contig_flag | (self.flags_ & USM_ARRAY_WRITABLE))
            self.nd_ = new_nd
            self.shape_ = shape_ptr
            self.strides_ = strides_ptr
        else:
            raise InternalUSMArrayError(
                "Encountered in shape setter, error code {err}".format(err)
            )

    @property
    def strides(self):
        """
        Returns memory displacement in array elements, upon unit
        change of respective index.

        For example, for strides ``(s1, s2, s3)`` and multi-index
        ``(i1, i2, i3)`` position of the respective element relative
        to zero multi-index element is ``s1*s1 + s2*i2 + s3*i3``.

        :Example:

            .. code-block:: python

                from dpctl import tensor

                x = tensor.zeros((20, 30))
                xv = x[10:, :15]

                multi_id = (3, 5)
                byte_displacement = xv[multi_id]._pointer - xv[0, 0]._pointer
                element_displacement = sum(
                    i * s for i, s in zip(multi_id, xv.strides)
                )
                assert byte_displacement == element_displacement * xv.itemsize
        """
        if (self.strides_):
            return _make_int_tuple(self.nd_, self.strides_)
        else:
            if (self.flags_ & USM_ARRAY_C_CONTIGUOUS):
                return _c_contig_strides(self.nd_, self.shape_)
            elif (self.flags_ & USM_ARRAY_F_CONTIGUOUS):
                return _f_contig_strides(self.nd_, self.shape_)
            else:
                raise ValueError("Inconsistent usm_ndarray data")

    @property
    def flags(self):
        """
        Returns :class:`dpctl.tensor._flags.Flags` object.
        """
        return _flags.Flags(self, self.flags_)

    cdef _set_writable_flag(self, int flag):
        cdef int mask = (USM_ARRAY_WRITABLE if flag else 0)
        self.flags_ = _copy_writable(self.flags_, mask)

    @property
    def usm_type(self):
        """
        USM type of underlying memory. Possible values are:

            * ``"device"``
                USM-device allocation in device memory, only accessible
                to kernels executed on the device
            * ``"shared"``
                USM-shared allocation in device memory, accessible both
                from the device and from host
            * ``"host"``
                USM-host allocation in host memory, accessible both
                from the device and from host

        See: https://docs.oneapi.com/versions/latest/dpcpp/iface/usm.html
        """
        return self.base_.get_usm_type()

    @property
    def itemsize(self):
        """
        Size of array element in bytes.
        """
        return self.get_itemsize()

    @property
    def nbytes(self):
        """
        Total bytes consumed by the elements of the array.
        """
        return (
            shape_to_elem_count(self.nd_, self.shape_) *
            self.get_itemsize())

    @property
    def size(self):
        """
        Number of elements in the array.
        """
        return shape_to_elem_count(self.nd_, self.shape_)

    @property
    def dtype(self):
        """
        Returns NumPy's dtype corresponding to the type of the array elements.
        """
        return np.dtype(_make_typestr(self.typenum_))

    @property
    def sycl_queue(self):
        """
        Returns :class:`dpctl.SyclQueue` object associated with USM data.
        """
        return self.get_sycl_queue()

    @property
    def sycl_device(self):
        """
        Returns :class:`dpctl.SyclDevice` object on which USM data
        was allocated.
        """
        q = self.sycl_queue
        return q.sycl_device

    @property
    def device(self):
        """
        Returns :class:`dpctl.tensor.Device` object representing
        residence of the array data.

        The ``Device`` object represents Array API notion of the
        device, and contains :class:`dpctl.SyclQueue` associated
        with this array. Hence, ``.device`` property provides
        information distinct from ``.sycl_device`` property.

        :Example:

            .. code-block:: python

                >>> from dpctl import tensor
                >>> x = tensor.ones(10)
                >>> x.device
                Device(level_zero:gpu:0)
        """
        return Device.create_device(self.sycl_queue)

    @property
    def sycl_context(self):
        """
        Returns :class:`dpctl.SyclContext` object to which USM data is bound.
        """
        q = self.sycl_queue
        return q.sycl_context

    @property
    def T(self):
        """Returns transposed array for 2D array, raises ``ValueError``
        otherwise.
        """
        if self.nd_ == 2:
            return _transpose(self)
        else:
            raise ValueError(
                "array.T requires array to have 2 dimensions. "
                "Use array.mT to transpose stacks of matrices and "
                "dpctl.tensor.permute_dims() to permute dimensions."
            )

    @property
    def mT(self):
        """ Returns array (a view) where the last two dimensions are
        transposed.
        """
        if self.nd_ < 2:
            raise ValueError(
                "array.mT requires array to have at least 2 dimensions."
            )
        return _m_transpose(self)

    @property
    def real(self):
        """
        Returns view into real component for arrays with
        complex data-types and returns itself for all other
        data-types.

        :Example:

            .. code-block:: python

                from dpctl import tensor

                # Create complex array from
                # arrays of real and imaginary parts

                re = tensor.linspace(-1, 1, num=100, dtype="f4")
                im = tensor.full_like(re, fill_value=tensor.pi)

                z = tensor.empty_like(re, dtype="c8")
                z.real[:] = re
                z.imag[:] = im
        """
        # explicitly check for UAR_HALF, which is greater than UAR_CFLOAT
        if (self.typenum_ < UAR_CFLOAT or self.typenum_ == UAR_HALF):
            # elements are real
            return self
        if (self.typenum_ < UAR_TYPE_SENTINEL):
            return _real_view(self)

    @property
    def imag(self):
        """ Returns view into imaginary component for arrays with
        complex data-types and returns new zero array for all other
        data-types.

        :Example:

            .. code-block:: python

                from dpctl import tensor

                # Reset imaginary part of complex array

                z = tensor.ones(100, dtype="c8")
                z.imag[:] = dpt.pi/2
        """
        # explicitly check for UAR_HALF, which is greater than UAR_CFLOAT
        if (self.typenum_ < UAR_CFLOAT or self.typenum_ == UAR_HALF):
            # elements are real
            return _zero_like(self)
        if (self.typenum_ < UAR_TYPE_SENTINEL):
            return _imag_view(self)

    def __getitem__(self, ind):
        cdef tuple _meta = _basic_slice_meta(
            ind, (<object>self).shape, (<object> self).strides,
            self.get_offset())
        cdef usm_ndarray res
        cdef int i = 0
        cdef bint matching = 1

        if len(_meta) < 5:
            raise RuntimeError

        res = usm_ndarray.__new__(
            usm_ndarray,
            _meta[0],
            dtype=_make_typestr(self.typenum_),
            strides=_meta[1],
            buffer=self.base_,
            offset=_meta[2]
        )
        res.array_namespace_ = self.array_namespace_

        adv_ind = _meta[3]
        adv_ind_start_p = _meta[4]

        if adv_ind_start_p < 0:
            res.flags_ = _copy_writable(res.flags_, self.flags_)
            return res

        from ._copy_utils import _extract_impl, _nonzero_impl, _take_multi_index

        # if len(adv_ind == 1), the (only) element is always an array
        if len(adv_ind) == 1 and adv_ind[0].dtype == dpt_bool:
            key_ = adv_ind[0]
            adv_ind_end_p = key_.ndim + adv_ind_start_p
            if adv_ind_end_p > res.ndim:
                raise IndexError("too many indices for the array")
            key_shape = key_.shape
            arr_shape = res.shape[adv_ind_start_p:adv_ind_end_p]
            for i in range(key_.ndim):
                if matching:
                    if not key_shape[i] == arr_shape[i] and key_shape[i] > 0:
                        matching = 0
            if not matching:
                raise IndexError(
                    "boolean index did not match indexed array in dimensions"
                )
            res = _extract_impl(res, key_, axis=adv_ind_start_p)
            res.flags_ = _copy_writable(res.flags_, self.flags_)
            return res

        if any(
            (
                isinstance(ind, usm_ndarray) and ind.dtype == dpt_bool
            ) for ind in adv_ind
        ):
            adv_ind_int = list()
            for ind in adv_ind:
                if isinstance(ind, usm_ndarray) and ind.dtype == dpt_bool:
                    adv_ind_int.extend(_nonzero_impl(ind))
                else:
                    adv_ind_int.append(ind)
            res = _take_multi_index(res, tuple(adv_ind_int), adv_ind_start_p)
            res.flags_ = _copy_writable(res.flags_, self.flags_)
            return res

        res = _take_multi_index(res, adv_ind, adv_ind_start_p)
        res.flags_ = _copy_writable(res.flags_, self.flags_)
        return res

    def to_device(self, target_device, /, *, stream=None):
        """ to_device(target_device, /, *, stream=None)

        Transfers this array to specified target device.

        :Example:
            .. code-block:: python

                import dpctl
                import dpctl.tensor as dpt

                x = dpt.full(10**6, 2, dtype="int64")
                q_prof = dpctl.SyclQueue(
                    x.sycl_device, property="enable_profiling")
                # return a view with profile-enabled queue
                y = x.to_device(q_prof)
                timer = dpctl.SyclTimer()
                with timer(q_prof):
                    z = y * y
                print(timer.dt)

        Args:
            target_device (object):
                Array API concept of target device.
                It can be a oneAPI filter selector string,
                an instance of :class:`dpctl.SyclDevice` corresponding to a
                non-partitioned SYCL device, an instance of
                :class:`dpctl.SyclQueue`, or a :class:`dpctl.tensor.Device`
                object returned by :attr:`dpctl.tensor.usm_ndarray.device`.
            stream (:class:`dpctl.SyclQueue`, optional):
                Execution queue to synchronize with. If ``None``,
                synchronization is not performed.

        Returns:
            usm_ndarray:
                A view if data copy is not required, and a copy otherwise.
                If copying is required, it is done by copying from the original
                allocation device to the host, followed by copying from host
                to the target device.
        """
        cdef c_dpctl.DPCTLSyclQueueRef QRef = NULL
        cdef c_dpmem._Memory arr_buf
        d = Device.create_device(target_device)

        _validate_and_use_stream(stream, self.sycl_queue)

        if (d.sycl_context == self.sycl_context):
            arr_buf = <c_dpmem._Memory> self.usm_data
            QRef = (<c_dpctl.SyclQueue> d.sycl_queue).get_queue_ref()
            view_buffer = c_dpmem._Memory.create_from_usm_pointer_size_qref(
                <DPCTLSyclUSMRef>arr_buf.get_data_ptr(),
                arr_buf.nbytes,
                QRef,
                memory_owner=arr_buf
            )
            res = usm_ndarray(
                self.shape,
                self.dtype,
                buffer=view_buffer,
                strides=self.strides,
                offset=self.get_offset()
            )
            res.flags_ = self.flags_
            return res
        else:
            nbytes = self.usm_data.nbytes
            copy_buffer = type(self.usm_data)(
                nbytes, queue=d.sycl_queue
            )
            copy_buffer.copy_from_device(self.usm_data)
            res = usm_ndarray(
                self.shape,
                self.dtype,
                buffer=copy_buffer,
                strides=self.strides,
                offset=self.get_offset()
            )
            res.flags_ = self.flags_
            return res

    def _set_namespace(self, mod):
        """ Sets array namespace to given module `mod`. """
        self.array_namespace_ = mod

    def __array_namespace__(self, api_version=None):
        """
        Returns array namespace, member functions of which
        implement data API.

        Args:
            api_version (str, optional)
                Request namespace compliant with given version of
                array API. If ``None``, namespace for the most
                recent supported version is returned.
                Default: ``None``.
        """
        if api_version is not None:
            from ._array_api import __array_api_version__
            if not isinstance(api_version, str):
                raise TypeError(f"Expected type str, got {type(api_version)}")
            if api_version != __array_api_version__:
                raise ValueError(f"Only {__array_api_version__} is supported")
        return (
            self.array_namespace_
            if self.array_namespace_ is not None
            else dpctl.tensor
        )

    def __bool__(self):
        if self.size == 1:
            _check_0d_scalar_conversion(self)
            view = _as_zero_dim_ndarray(self)
            return view.__bool__()

        if self.size == 0:
            raise ValueError(
                "The truth value of an empty array is ambiguous"
            )

        raise ValueError(
            "The truth value of an array with more than one element is "
            "ambiguous. Use dpctl.tensor.any() or dpctl.tensor.all()"
        )

    def __float__(self):
        if self.size == 1:
            _check_0d_scalar_conversion(self)
            view = _as_zero_dim_ndarray(self)
            return view.__float__()

        raise ValueError(
            "only size-1 arrays can be converted to Python scalars"
        )

    def __complex__(self):
        if self.size == 1:
            _check_0d_scalar_conversion(self)
            view = _as_zero_dim_ndarray(self)
            return view.__complex__()

        raise ValueError(
            "only size-1 arrays can be converted to Python scalars"
        )

    def __int__(self):
        if self.size == 1:
            _check_0d_scalar_conversion(self)
            view = _as_zero_dim_ndarray(self)
            return view.__int__()

        raise ValueError(
            "only size-1 arrays can be converted to Python scalars"
        )

    def __index__(self):
        if np.issubdtype(self.dtype, np.integer):
            return int(self)

        raise IndexError("only integer arrays are valid indices")

    def __abs__(self):
        return dpctl.tensor.abs(self)

    def __add__(self, other):
        """
        Implementation for operator.add
        """
        return dpctl.tensor.add(self, other)

    def __and__(self, other):
        "Implementation for operator.and"
        return dpctl.tensor.bitwise_and(self, other)

    def __dlpack__(
        self, *, stream=None, max_version=None, dl_device=None, copy=None
    ):
        """
        Produces DLPack capsule.

        Args:
            stream (:class:`dpctl.SyclQueue`, optional):
                Execution queue to synchronize with.
                If ``None``, synchronization is not performed.
                Default: ``None``.
            max_version (tuple[int, int], optional):
                The maximum DLPack version the consumer (caller of
                ``__dlpack__``) supports. As ``__dlpack__`` may not
                always return a DLPack capsule with version
                `max_version`, the consumer must verify the version
                even if this argument is passed.
                Default: ``None``.
            dl_device (tuple[enum.Enum, int], optional):
                The device the returned DLPack capsule will be
                placed on.
                The device must be a 2-tuple matching the format of
                ``__dlpack_device__`` method, an integer enumerator
                representing the device type followed by an integer
                representing the index of the device.
                Default: ``None``.
            copy (bool, optional):
                Boolean indicating whether or not to copy the input.

                * If ``copy`` is ``True``, the input will always be
                  copied.
                * If ``False``, a ``BufferError`` will be raised if a
                  copy is deemed necessary.
                * If ``None``, a copy will be made only if deemed
                  necessary, otherwise, the existing memory buffer will
                  be reused.

                Default: ``None``.

        Raises:
            MemoryError:
                when host memory can not be allocated.
            DLPackCreationError:
                when array is allocated on a partitioned
                SYCL device, or with a non-default context.
            BufferError:
                when a copy is deemed necessary but ``copy``
                is ``False`` or when the provided ``dl_device``
                cannot be handled.
        """
        if max_version is None:
            # legacy path for DLManagedTensor
            # copy kwarg ignored because copy flag can't be set
            _caps = c_dlpack.to_dlpack_capsule(self)
            _validate_and_use_stream(stream, self.sycl_queue)
            return _caps
        else:
            if not isinstance(max_version, tuple) or len(max_version) != 2:
                raise TypeError(
                    "`__dlpack__` expects `max_version` to be a "
                    "2-tuple of integers `(major, minor)`, instead "
                    f"got {max_version}"
                )
            dpctl_dlpack_version = get_build_dlpack_version()
            if max_version[0] >= dpctl_dlpack_version[0]:
                # DLManagedTensorVersioned path
                if dl_device is not None:
                    if not isinstance(dl_device, tuple) or len(dl_device) != 2:
                        raise TypeError(
                            "`__dlpack__` expects `dl_device` to be a 2-tuple "
                            "of `(device_type, device_id)`, instead "
                            f"got {dl_device}"
                        )
                    if dl_device != self.__dlpack_device__():
                        if copy is False:
                            raise BufferError(
                                "array cannot be placed on the requested "
                                "device without a copy"
                            )
                        if _is_host_cpu(dl_device):
                            if stream is not None:
                                raise ValueError(
                                    "`stream` must be `None` when `dl_device` "
                                    "is of type `kDLCPU`"
                                )
                            from ._copy_utils import _copy_to_numpy
                            _arr = _copy_to_numpy(self)
                            _arr.flags["W"] = self.flags["W"]
                            return c_dlpack.numpy_to_dlpack_versioned_capsule(
                                _arr, True
                            )
                        else:
                            raise BufferError(
                                f"targeting `dl_device` {dl_device} with "
                                "`__dlpack__` is not yet implemented"
                            )
                if copy is None:
                    copy = False
                # TODO: strategy for handling stream on different device
                # from dl_device
                if copy:
                    _validate_and_use_stream(stream, self.sycl_queue)
                    nbytes = self.usm_data.nbytes
                    copy_buffer = type(self.usm_data)(
                        nbytes, queue=self.sycl_queue
                    )
                    copy_buffer.copy_from_device(self.usm_data)
                    _copied_arr = usm_ndarray(
                        self.shape,
                        self.dtype,
                        buffer=copy_buffer,
                        strides=self.strides,
                        offset=self.get_offset()
                    )
                    _copied_arr.flags_ = self.flags_
                    _caps = c_dlpack.to_dlpack_versioned_capsule(
                        _copied_arr, copy
                    )
                else:
                    _caps = c_dlpack.to_dlpack_versioned_capsule(self, copy)
                    _validate_and_use_stream(stream, self.sycl_queue)
                return _caps
            else:
                # legacy path for DLManagedTensor
                _caps = c_dlpack.to_dlpack_capsule(self)
                _validate_and_use_stream(stream, self.sycl_queue)
                return _caps

    def __dlpack_device__(self):
        """
        Gives a tuple (``device_type``, ``device_id``) corresponding to
        ``DLDevice`` entry in ``DLTensor`` in DLPack protocol.

        The tuple describes the non-partitioned device where the array has been
        allocated, or the non-partitioned parent device of the allocation
        device.

        See :class:`dpctl.tensor.DLDeviceType` for a list of devices supported
        by the DLPack protocol.

        Raises:
            DLPackCreationError:
                when the ``device_id`` could not be determined.
        """
        try:
            dev_id = self.sycl_device.get_device_id()
        except ValueError as e:
            raise c_dlpack.DLPackCreationError(
                "Could not determine id of the device where array was "
                "allocated."
            )
        return (
            DLDeviceType.kDLOneAPI,
            dev_id,
        )

    def __eq__(self, other):
        return dpctl.tensor.equal(self, other)

    def __floordiv__(self, other):
        return dpctl.tensor.floor_divide(self, other)

    def __ge__(self, other):
        return dpctl.tensor.greater_equal(self, other)

    def __gt__(self, other):
        return dpctl.tensor.greater(self, other)

    def __invert__(self):
        return dpctl.tensor.bitwise_invert(self)

    def __le__(self, other):
        return dpctl.tensor.less_equal(self, other)

    def __len__(self):
        if (self.nd_):
            return self.shape[0]
        else:
            raise TypeError("len() of unsized object")

    def __lshift__(self, other):
        return dpctl.tensor.bitwise_left_shift(self, other)

    def __lt__(self, other):
        return dpctl.tensor.less(self, other)

    def __matmul__(self, other):
        return dpctl.tensor.matmul(self, other)

    def __mod__(self, other):
        return dpctl.tensor.remainder(self, other)

    def __mul__(self, other):
        return dpctl.tensor.multiply(self, other)

    def __ne__(self, other):
        return dpctl.tensor.not_equal(self, other)

    def __neg__(self):
        return dpctl.tensor.negative(self)

    def __or__(self, other):
        return dpctl.tensor.bitwise_or(self, other)

    def __pos__(self):
        return dpctl.tensor.positive(self)

    def __pow__(self, other):
        return dpctl.tensor.pow(self, other)

    def __rshift__(self, other):
        return dpctl.tensor.bitwise_right_shift(self, other)

    def __setitem__(self, key, rhs):
        cdef tuple _meta
        cdef usm_ndarray Xv

        if (self.flags_ & USM_ARRAY_WRITABLE) == 0:
            raise ValueError("Can not modify read-only array.")

        _meta = _basic_slice_meta(
            key, (<object>self).shape, (<object> self).strides,
            self.get_offset()
        )

        if len(_meta) < 5:
            raise RuntimeError

        Xv = usm_ndarray.__new__(
            usm_ndarray,
            _meta[0],
            dtype=_make_typestr(self.typenum_),
            strides=_meta[1],
            buffer=self.base_,
            offset=_meta[2],
        )
        # set namespace
        Xv.array_namespace_ = self.array_namespace_

        from ._copy_utils import (
            _copy_from_numpy_into,
            _copy_from_usm_ndarray_to_usm_ndarray,
            _nonzero_impl,
            _place_impl,
            _put_multi_index,
        )

        adv_ind = _meta[3]
        adv_ind_start_p = _meta[4]

        if adv_ind_start_p < 0:
            # basic slicing
            if isinstance(rhs, usm_ndarray):
                _copy_from_usm_ndarray_to_usm_ndarray(Xv, rhs)
            else:
                if hasattr(rhs, "__sycl_usm_array_interface__"):
                    from dpctl.tensor import asarray
                    try:
                        rhs_ar = asarray(rhs)
                        _copy_from_usm_ndarray_to_usm_ndarray(Xv, rhs_ar)
                    except Exception:
                        raise ValueError(
                            f"Input of type {type(rhs)} could not be "
                            "converted to usm_ndarray"
                        )
                else:
                    rhs_np = np.asarray(rhs)
                    if type_bytesize(rhs_np.dtype.num) < 0:
                        raise ValueError(
                            f"Input of type {type(rhs)} can not be "
                            "assigned to usm_ndarray because of "
                            f"unsupported data type '{rhs_np.dtype}'"
                        )
                    try:
                        _copy_from_numpy_into(Xv, rhs_np)
                    except Exception:
                        raise ValueError(
                            f"Input of type {type(rhs)} could not be "
                            "copied into dpctl.tensor.usm_ndarray"
                        )
            return

        if len(adv_ind) == 1 and adv_ind[0].dtype == dpt_bool:
            _place_impl(Xv, adv_ind[0], rhs, axis=adv_ind_start_p)
            return

        if any(
            (
                isinstance(ind, usm_ndarray) and ind.dtype == dpt_bool
            ) for ind in adv_ind
        ):
            adv_ind_int = list()
            for ind in adv_ind:
                if isinstance(ind, usm_ndarray) and ind.dtype == dpt_bool:
                    adv_ind_int.extend(_nonzero_impl(ind))
                else:
                    adv_ind_int.append(ind)
            _put_multi_index(Xv, tuple(adv_ind_int), adv_ind_start_p, rhs)
            return

        _put_multi_index(Xv, adv_ind, adv_ind_start_p, rhs)
        return

    def __sub__(self, other):
        return dpctl.tensor.subtract(self, other)

    def __truediv__(self, other):
        return dpctl.tensor.divide(self, other)

    def __xor__(self, other):
        return dpctl.tensor.bitwise_xor(self, other)

    def __radd__(self, other):
        return dpctl.tensor.add(other, self)

    def __rand__(self, other):
        return dpctl.tensor.bitwise_and(other, self)

    def __rfloordiv__(self, other):
        return dpctl.tensor.floor_divide(other, self)

    def __rlshift__(self, other):
        return dpctl.tensor.bitwise_left_shift(other, self)

    def __rmatmul__(self, other):
        return dpctl.tensor.matmul(other, self)

    def __rmod__(self, other):
        return dpctl.tensor.remainder(other, self)

    def __rmul__(self, other):
        return dpctl.tensor.multiply(other, self)

    def __ror__(self, other):
        return dpctl.tensor.bitwise_or(other, self)

    def __rpow__(self, other):
        return dpctl.tensor.pow(other, self)

    def __rrshift__(self, other):
        return dpctl.tensor.bitwise_right_shift(other, self)

    def __rsub__(self, other):
        return dpctl.tensor.subtract(other, self)

    def __rtruediv__(self, other):
        return dpctl.tensor.divide(other, self)

    def __rxor__(self, other):
        return dpctl.tensor.bitwise_xor(other, self)

    def __iadd__(self, other):
        return dpctl.tensor.add._inplace_op(self, other)

    def __iand__(self, other):
        return dpctl.tensor.bitwise_and._inplace_op(self, other)

    def __ifloordiv__(self, other):
        return dpctl.tensor.floor_divide._inplace_op(self, other)

    def __ilshift__(self, other):
        return dpctl.tensor.bitwise_left_shift._inplace_op(self, other)

    def __imatmul__(self, other):
        return dpctl.tensor.matmul(self, other, out=self, dtype=self.dtype)

    def __imod__(self, other):
        return dpctl.tensor.remainder._inplace_op(self, other)

    def __imul__(self, other):
        return dpctl.tensor.multiply._inplace_op(self, other)

    def __ior__(self, other):
        return dpctl.tensor.bitwise_or._inplace_op(self, other)

    def __ipow__(self, other):
        return dpctl.tensor.pow._inplace_op(self, other)

    def __irshift__(self, other):
        return dpctl.tensor.bitwise_right_shift._inplace_op(self, other)

    def __isub__(self, other):
        return dpctl.tensor.subtract._inplace_op(self, other)

    def __itruediv__(self, other):
        return dpctl.tensor.divide._inplace_op(self, other)

    def __ixor__(self, other):
        return dpctl.tensor.bitwise_xor._inplace_op(self, other)

    def __str__(self):
        return usm_ndarray_str(self)

    def __repr__(self):
        return usm_ndarray_repr(self)

    def __array__(self, dtype=None, /, *, copy=None):
        """NumPy's array protocol method to disallow implicit conversion.

        Without this definition, `numpy.asarray(usm_ar)` converts
        usm_ndarray instance into NumPy array with data type `object`
        and every element being 0d usm_ndarray.

        https://github.com/IntelPython/dpctl/pull/1384#issuecomment-1707212972
        """
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Use `dpctl.tensor.asnumpy` to copy data from this "
            "`dpctl.tensor.usm_ndarray` instance to NumPy array"
        )


cdef usm_ndarray _real_view(usm_ndarray ary):
    """
    View into real parts of a complex type array
    """
    cdef int r_typenum_ = -1
    cdef usm_ndarray r = None
    cdef Py_ssize_t offset_elems = 0

    if (ary.typenum_ == UAR_CFLOAT):
        r_typenum_ = UAR_FLOAT
    elif (ary.typenum_ == UAR_CDOUBLE):
        r_typenum_ = UAR_DOUBLE
    else:
        raise InternalUSMArrayError(
            "_real_view call on array of non-complex type.")

    offset_elems = ary.get_offset() * 2
    r = usm_ndarray.__new__(
        usm_ndarray,
        _make_int_tuple(ary.nd_, ary.shape_) if ary.nd_ > 0 else tuple(),
        dtype=_make_typestr(r_typenum_),
        strides=tuple(2 * si for si in ary.strides),
        buffer=ary.base_,
        offset=offset_elems,
        order=("C" if (ary.flags_ & USM_ARRAY_C_CONTIGUOUS) else "F")
    )
    r.flags_ = _copy_writable(r.flags_, ary.flags_)
    r.array_namespace_ = ary.array_namespace_
    return r


cdef usm_ndarray _imag_view(usm_ndarray ary):
    """
    View into imaginary parts of a complex type array
    """
    cdef int r_typenum_ = -1
    cdef usm_ndarray r = None
    cdef Py_ssize_t offset_elems = 0

    if (ary.typenum_ == UAR_CFLOAT):
        r_typenum_ = UAR_FLOAT
    elif (ary.typenum_ == UAR_CDOUBLE):
        r_typenum_ = UAR_DOUBLE
    else:
        raise InternalUSMArrayError(
            "_imag_view call on array of non-complex type.")

    # displace pointer to imaginary part
    offset_elems = 2 * ary.get_offset() + 1
    r = usm_ndarray.__new__(
        usm_ndarray,
        _make_int_tuple(ary.nd_, ary.shape_) if ary.nd_ > 0 else tuple(),
        dtype=_make_typestr(r_typenum_),
        strides=tuple(2 * si for si in ary.strides),
        buffer=ary.base_,
        offset=offset_elems,
        order=("C" if (ary.flags_ & USM_ARRAY_C_CONTIGUOUS) else "F")
    )
    r.flags_ = _copy_writable(r.flags_, ary.flags_)
    r.array_namespace_ = ary.array_namespace_
    return r


cdef usm_ndarray _transpose(usm_ndarray ary):
    """
    Construct transposed array without copying the data
    """
    cdef usm_ndarray r = usm_ndarray.__new__(
        usm_ndarray,
        _make_reversed_int_tuple(ary.nd_, ary.shape_),
        dtype=_make_typestr(ary.typenum_),
        strides=(
            _make_reversed_int_tuple(ary.nd_, ary.strides_)
            if (ary.strides_) else None),
        buffer=ary.base_,
        order=("F" if (ary.flags_ & USM_ARRAY_C_CONTIGUOUS) else "C"),
        offset=ary.get_offset()
    )
    r.flags_ = _copy_writable(r.flags_, ary.flags_)
    return r


cdef usm_ndarray _m_transpose(usm_ndarray ary):
    """
    Construct matrix transposed array
    """
    cdef usm_ndarray r = usm_ndarray.__new__(
        usm_ndarray,
        _swap_last_two(_make_int_tuple(ary.nd_, ary.shape_)),
        dtype=_make_typestr(ary.typenum_),
        strides=_swap_last_two(ary.strides),
        buffer=ary.base_,
        order=("F" if (ary.flags_ & USM_ARRAY_C_CONTIGUOUS) else "C"),
        offset=ary.get_offset()
    )
    r.flags_ = _copy_writable(r.flags_, ary.flags_)
    return r


cdef usm_ndarray _zero_like(usm_ndarray ary):
    """
    Make C-contiguous array of zero elements with same shape,
    type, device, and sycl_queue as ary.
    """
    cdef dt = _make_typestr(ary.typenum_)
    cdef usm_ndarray r = usm_ndarray(
        _make_int_tuple(ary.nd_, ary.shape_) if ary.nd_ > 0 else tuple(),
        dtype=dt,
        buffer=ary.base_.get_usm_type(),
        buffer_ctor_kwargs={"queue": ary.get_sycl_queue()},
    )
    r.base_.memset()
    return r


cdef api char* UsmNDArray_GetData(usm_ndarray arr):
    """Get allocation pointer of zero index element of array """
    return arr.get_data()


cdef api int UsmNDArray_GetNDim(usm_ndarray arr):
    """Get array rank: length of its shape"""
    return arr.get_ndim()


cdef api Py_ssize_t* UsmNDArray_GetShape(usm_ndarray arr):
    """Get host pointer to shape vector"""
    return arr.get_shape()


cdef api Py_ssize_t* UsmNDArray_GetStrides(usm_ndarray arr):
    """Get host pointer to strides vector"""
    return arr.get_strides()


cdef api int UsmNDArray_GetTypenum(usm_ndarray arr):
    """Get type number for data type of array elements"""
    return arr.get_typenum()


cdef api int UsmNDArray_GetElementSize(usm_ndarray arr):
    """Get array element size in bytes"""
    return arr.get_itemsize()


cdef api int UsmNDArray_GetFlags(usm_ndarray arr):
    """Get flags of array"""
    return arr.get_flags()


cdef api c_dpctl.DPCTLSyclQueueRef UsmNDArray_GetQueueRef(usm_ndarray arr):
    """Get DPCTLSyclQueueRef for queue associated with the array"""
    return arr.get_queue_ref()


cdef api Py_ssize_t UsmNDArray_GetOffset(usm_ndarray arr):
    """Get offset of zero-index array element from the beginning of the USM
    allocation"""
    return arr.get_offset()


cdef api object UsmNDArray_GetUSMData(usm_ndarray arr):
    """Get USM data object underlying the array"""
    return arr.get_base()


cdef api void UsmNDArray_SetWritableFlag(usm_ndarray arr, int flag):
    """Set/unset USM_ARRAY_WRITABLE in the given array `arr`."""
    arr._set_writable_flag(flag)


cdef api object UsmNDArray_MakeSimpleFromMemory(
    int nd, const Py_ssize_t *shape, int typenum,
    c_dpmem._Memory mobj, Py_ssize_t offset, char order
):
    """Create contiguous usm_ndarray.

    Args:
        nd: number of dimensions (non-negative)
        shape: array of nd non-negative array's sizes along each dimension
        typenum: array elemental type number
        ptr: pointer to the start of allocation
        QRef: DPCTLSyclQueueRef associated with the allocation
        offset: distance between element with zero multi-index and the
                start of allocation
        order: Memory layout of the array. Use 'C' for C-contiguous or
               row-major layout; 'F' for F-contiguous or column-major layout
    Returns:
        Created usm_ndarray instance
    """
    cdef object shape_tuple = _make_int_tuple(nd, <Py_ssize_t *>shape)
    cdef usm_ndarray arr = usm_ndarray(
        shape_tuple,
        dtype=_make_typestr(typenum),
        buffer=mobj,
        offset=offset,
        order=<bytes>(order)
    )
    return arr


cdef api object UsmNDArray_MakeSimpleFromPtr(
    size_t nelems,
    int typenum,
    c_dpctl.DPCTLSyclUSMRef ptr,
    c_dpctl.DPCTLSyclQueueRef QRef,
    object owner
):
    """Create 1D contiguous usm_ndarray from pointer.

    Args:
        nelems: number of elements in array
        typenum: array elemental type number
        ptr: pointer to the start of allocation
        QRef: DPCTLSyclQueueRef associated with the allocation
        owner: Python object managing lifetime of USM allocation.
               Value None implies transfer of USM allocation ownership
               to the created array object.
    Returns:
        Created usm_ndarray instance
    """
    cdef int itemsize = type_bytesize(typenum)
    if (itemsize < 1):
        raise ValueError(
            "dtype with typenum=" + str(typenum) + " is not supported."
        )
    cdef size_t nbytes = (<size_t> itemsize) * nelems
    cdef c_dpmem._Memory mobj
    mobj = c_dpmem._Memory.create_from_usm_pointer_size_qref(
        ptr, nbytes, QRef, memory_owner=owner
    )
    cdef usm_ndarray arr = usm_ndarray(
        (nelems,),
        dtype=_make_typestr(typenum),
        buffer=mobj
    )
    return arr

cdef api object UsmNDArray_MakeFromPtr(
    int nd,
    const Py_ssize_t *shape,
    int typenum,
    const Py_ssize_t *strides,
    c_dpctl.DPCTLSyclUSMRef ptr,
    c_dpctl.DPCTLSyclQueueRef QRef,
    Py_ssize_t offset,
    object owner
):
    """
    General usm_ndarray constructor from externally made USM-allocation.

    Args:
        nd: number of dimensions (non-negative)
        shape: array of nd non-negative array's sizes along each dimension
        typenum: array elemental type number
        strides: array of nd strides along each dimension in elements
        ptr: pointer to the start of allocation
        QRef: DPCTLSyclQueueRef associated with the allocation
        offset: distance between element with zero multi-index and the
                start of allocation
        owner: Python object managing lifetime of USM allocation.
               Value None implies transfer of USM allocation ownership
               to the created array object.
    Returns:
        Created usm_ndarray instance
    """
    cdef int itemsize = type_bytesize(typenum)
    cdef size_t nelems = 1
    cdef Py_ssize_t min_disp = 0
    cdef Py_ssize_t max_disp = 0
    cdef Py_ssize_t step_ = 0
    cdef Py_ssize_t dim_ = 0
    cdef it = 0
    cdef c_dpmem._Memory mobj
    cdef usm_ndarray arr
    cdef object obj_shape
    cdef object obj_strides

    if (itemsize < 1):
        raise ValueError(
            "dtype with typenum=" + str(typenum) + " is not supported."
        )
    if (nd < 0):
        raise ValueError("Dimensionality must be non-negative")
    if (ptr is NULL or QRef is NULL):
        raise ValueError(
            "Non-null USM allocation pointer and QRef are expected"
        )
    if (nd == 0):
        # case of 0d scalars
        mobj = c_dpmem._Memory.create_from_usm_pointer_size_qref(
            ptr, itemsize, QRef, memory_owner=owner
        )
        arr = usm_ndarray(
            tuple(),
            dtype=_make_typestr(typenum),
            buffer=mobj
        )
        return arr
    if (shape is NULL or strides is NULL):
        raise ValueError("Both shape and stride vectors are required")
    for it in range(nd):
        dim_ = shape[it]
        if dim_ < 0:
            raise ValueError(
                f"Dimension along axis {it} must be non-negative"
            )
        nelems *= dim_
        if dim_ > 0:
            step_ = strides[it]
            if step_ > 0:
                max_disp += step_ * (dim_ - 1)
            else:
                min_disp += step_ * (dim_ - 1)

    obj_shape = _make_int_tuple(nd, shape)
    obj_strides = _make_int_tuple(nd, strides)
    if nelems == 0:
        mobj = c_dpmem._Memory.create_from_usm_pointer_size_qref(
            ptr, itemsize, QRef, memory_owner=owner
        )
        arr = usm_ndarray(
            obj_shape,
            dtype=_make_typestr(typenum),
            strides=obj_strides,
            buffer=mobj,
            offset=0
        )
        return arr
    if offset + min_disp < 0:
        raise ValueError(
            "Given shape, strides and offset reference out-of-bound memory"
        )
    nbytes = (<size_t> itemsize) * (offset + max_disp + 1)
    mobj = c_dpmem._Memory.create_from_usm_pointer_size_qref(
        ptr, nbytes, QRef, memory_owner=owner
    )
    arr = usm_ndarray(
        obj_shape,
        dtype=_make_typestr(typenum),
        strides=obj_strides,
        buffer=mobj,
        offset=offset
    )
    return arr


def _is_object_with_buffer_protocol(o):
    "Returns True if object supports Python buffer protocol"
    return _is_buffer(o)
