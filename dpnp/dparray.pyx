# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
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


"""Module DPArray

This module contains Array class represents multi-dimensional array
using USB interface for an Intel GPU device.

"""


from libcpp cimport bool as cpp_bool

from dpnp.dpnp_iface_types import *
from dpnp.dpnp_iface import *
from dpnp.backend cimport *
from dpnp.dpnp_iface_statistics import min, max
from dpnp.dpnp_iface_logic import all, any
import numpy
cimport numpy

cimport dpnp.dpnp_utils as utils
cimport dpctl as c_dpctl
cimport dpctl.memory as c_dpctl_mem
import dpctl

cdef class _ArrayInterfaceData:
    cdef int nd
    cdef char * pointer
    cdef Py_ssize_t itemsize
    cdef int writeable
    cdef tuple shape
    cdef tuple strides
    cdef object dtype
    cdef c_dpctl.SyclQueue queue

    @staticmethod
    cdef _ArrayInterfaceData parse_sycl_usm_array_interface(object memory):
        cdef dict iface
        cdef _ArrayInterfaceData res_iface
        cdef sycl_ctx
        cdef char * iface_pointer
        
        if not hasattr(memory, "__sycl_usm_array_interface__"):
            raise TypeError("Memory object does not expose __sycl_usm_array_interface__")
        iface = memory.__sycl_usm_array_interface__
        iface_ver = iface['version']
        if (iface_ver != 1):
            raise TypeError("Memory object exposes __sycl_usm_array_interface__ of version {}, version 1 is expected".format(iface_ver))
        data_obj = iface.get('data', None)
        if (not isinstance(data_obj, (tuple, list)) or len(data_obj) != 2):
            raise TypeError("'data' entries in __sycl_usm_array_interface__ dictionary is not a tuple of length 2")
        
        iface_pointer = <char *>(<Py_ssize_t>data_obj[0]);
        writeable = (<int>data_obj[1])
        typestr = iface.get('typestr')
        dtype = numpy.dtype(typestr)
        if (dtype.kind == 'V'): # follow NumPy here
            typedescr = iface.get('descr', None)
            dtype = numpy.dtype((typestr, typedescr))
        shape = iface['shape']
        if not isinstance(shape, (list, tuple)):
            raise TypeError("'shape' entry in __sycl_usm_array_interface__ dictionary is not a list or a tuple")
        nd = len(shape)
        strides = iface.get('strides', None)
        if (strides is not None):
            if not isinstance(strides, (list, tuple)) or len(strides) != nd or nd == 0:
                raise TypeError("'shape' entry in __sycl_usm_array_interface__ dictionary is not a list or a tuple")
            
        offsets = iface.get('offsets', 0)
        syclobj = iface.get('syclobj')
        if not isinstance(syclobj, (dpctl.SyclQueue, dpctl.SyclContext)):
            raise TypeError("'syclobj' entry in __sycl_usm_array_interface__ dictionary is not "
                            "dpctl.SyclQueue or dpctl.SyclContext")
        if isinstance(syclobj, dpctl.SyclQueue):
            iface_queue = <c_dpctl.SyclQueue> syclobj
            usm_type = c_dpctl_mem._Memory.get_pointer_type(
                <c_dpctl.DPPLSyclUSMRef>iface_pointer, iface_queue.get_sycl_context())
            if (usm_type != b'shared'):
                raise TypeError("Expecting USM shared memory.")
        else:
            # Obtain device from pointer and context
            sycl_ctx = <c_dpctl.SyclContext> syclobj
            sycl_dev = c_dpctl_mem._Memory.get_pointer_device(<c_dpctl.DPPLSyclUSMRef>iface_pointer, ctx)
            usm_type = c_dpctl_mem._Memory.get_pointer_type(<c_dpctl.DPPLSyclUSMRef>iface_pointer, ctx)
            if (usm_type != b'shared'):
                raise TypeError("Expecting USM shared memory.")
            # Use context and device to create a queue to
            # be able to copy memory
            iface_queue = c_dpcl.SyclQueue._create_from_context_and_device(sycl_ctx, sycl_dev)


        res_iface = _ArrayInterfaceData.__new__(_ArrayInterfaceData)
        res_iface.pointer = iface_pointer
        res_iface.writeable = writeable
        res_iface.dtype = dtype
        res_iface.itemsize = dtype.itemsize
        res_iface.nd = nd
        res_iface.queue = iface_queue

        return res_iface


# initially copied from original
cdef class _flagsobj:
    aligned: bool
    updateifcopy: bool
    writeable: bool
    writebackifcopy: bool
    @property
    def behaved(self) -> bool: ...

    @property
    def c_contiguous(self) -> bool:
        return True

    @property
    def carray(self) -> bool: ...
    @property
    def contiguous(self) -> bool: ...

    @property
    def f_contiguous(self) -> bool:
        return False

    @property
    def farray(self) -> bool: ...
    @property
    def fnc(self) -> bool: ...
    @property
    def forc(self) -> bool: ...
    @property
    def fortran(self) -> bool: ...
    @property
    def num(self) -> int: ...

    @property
    def owndata(self) -> bool:
        return True

    def __getitem__(self, key: str) -> bool: ...
    def __setitem__(self, key: str, value: bool) -> None: ...


cdef class dparray:
    """Multi-dimensional array using USM interface for an Intel GPU device.

    This class implements a subset of methods of :class:`numpy.ndarray`.
    The difference is that this class allocates the array content using
    USM interface on the current GPU device.

    Args:
        shape (tuple of ints): Length of axes.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        memory: Python object exposing `__sycl_usm_array_interface__`` or None.
        strides (tuple of ints or None): Strides of data in memory.
        order ({'C', 'F'}): Row-major (C-style) or column-major (Fortran-style) order.

    Attributes:
        base (None or dpnp.dparray): Base array from which this array is created as a view.
        data (char *): Pointer to the array content head.
        ~dparray.dtype(numpy.dtype): Dtype object of elements type.

            .. seealso::
               `Data type objects (dtype) \
               <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`_
        ~dparray.size (int): Number of elements this array holds.

            This is equivalent to product over the shape tuple.

            .. seealso:: :attr:`numpy.ndarray.size`

    """

    def __init__(self, shape, dtype=float64, memory=None, strides=None, order=b'C'):
        cdef Py_ssize_t shape_it = 0
        cdef Py_ssize_t strides_it = 0
        cdef _ArrayInterfaceData iface_parsed

        if order != b'C':
            order = utils._normalize_order(order)

        if order != b'C' and order != b'F':
            raise TypeError(f"DPNP::__init__(): Parameter is not understood. order={order}")

        # dtype
        self._dparray_dtype = numpy.dtype(dtype)

        # size and shape
        cdef tuple shape_tup = utils._object_to_tuple(shape)
        self._dparray_shape.reserve(len(shape_tup))

        self._dparray_size = 1
        for shape_it in shape_tup:
            if shape_it < 0:
                raise ValueError("DPNP dparray::__init__(): Negative values in 'shape' are not allowed")
            # shape
            self._dparray_shape.push_back(shape_it)
            # size
            self._dparray_size *= shape_it

        # strides
        cdef tuple strides_tup = utils._object_to_tuple(strides)
        self._dparray_strides.reserve(len(strides_tup))
        for strides_it in strides_tup:
            if strides_it < 0:
                raise ValueError("DPNP dparray::__init__(): Negative values in 'strides' are not allowed")
            # stride
            self._dparray_strides.push_back(strides_it)

        # data
        if memory is None:
            self._dparray_data = dpnp_memory_alloc_c(self.nbytes)
            self.queue = c_dpctl.get_current_queue()
            self.base = None
        else:
            iface_parsed = _ArrayInterfaceData.parse_sycl_usm_array_interface(memory)
            if iface_parsed:
                self._dparray_data = iface_parsed.pointer
                self.queue = iface_parsed.queue
                self.base = memory
            

    def __dealloc__(self):
        """ Release owned memory

        """

        if (self.base is None):
            dpnp_memory_free_c(self._dparray_data)
        self.base = None


    def __repr__(self):
        """ Output information about the array to standard output

        Example:
          <DPNP DParray:name=dparray: mem=0x7ffad6fa4000: size=1048576: shape=[1024, 1024]: type=float64>

        """

        string = "<DPNP DParray:name={}".format(self.__class__.__name__)
        string += ": mem=0x{:x}".format( < size_t > self._dparray_data)
        string += ": size={}".format(self.size)
        string += ": shape={}".format(self.shape)
        string += ": type={}".format(self.dtype)
        string += ">"

        return string

    def __str__(self):
        """ Output values from the array to standard output

        Example:
          [[ 136.  136.  136.]
           [ 272.  272.  272.]
           [ 408.  408.  408.]]

        """

        for i in range(self.size):
            print(self[i], end=' ')

        return "<__str__ TODO>"

    
    # The definition order of attributes and methods are borrowed from the
    # order of documentation at the following NumPy document.
    # https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html

    # -------------------------------------------------------------------------
    # Memory layout
    # -------------------------------------------------------------------------

    @property
    def dtype(self):
        """Type of the elements in the array

        """

        return self._dparray_dtype

    @property
    def shape(self):
        """Lengths of axes. A tuple of numbers represents size of each dimention.

        Setter of this property involves reshaping without copy. If the array
        cannot be reshaped without copy, it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """

        return tuple(self._dparray_shape)

    @shape.setter
    def shape(self, newshape):
        """Set new lengths of axes. A tuple of numbers represents size of each dimention.
        It involves reshaping without copy. If the array cannot be reshaped without copy,
        it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """

        self._dparray_shape = newshape  # TODO strides, enpty dimentions and etc.

    @property
    def flags(self) -> _flagsobj:
        """Object containing memory-layout information.

        It only contains ``c_contiguous``, ``f_contiguous``, and ``owndata`` attributes.
        All of these are read-only. Accessing by indexes is also supported.

        .. seealso:: :attr:`numpy.ndarray.flags`

        """

        return _flagsobj()

    @property
    def strides(self):
        """Strides of axes in bytes.

        .. seealso:: :attr:`numpy.ndarray.strides`

        """
        if self._dparray_strides.empty():
            return None
        else:
            return tuple(self._dparray_strides)

    @property
    def ndim(self):
        """Number of dimensions.

        ``a.ndim`` is equivalent to ``len(a.shape)``.

        .. seealso:: :attr:`numpy.ndarray.ndim`

        """

        return self._dparray_shape.size()

    @property
    def itemsize(self):
        """Size of each element in bytes.

        .. seealso:: :attr:`numpy.ndarray.itemsize`

        """

        return self._dparray_dtype.itemsize

    @property
    def nbytes(self):
        """Total size of all elements in bytes.

        It does not count strides or alignment of elements.

        .. seealso:: :attr:`numpy.ndarray.nbytes`

        """

        return self.size * self.itemsize

    @property
    def size(self):
        """Number of elements in the array.

        .. seealso:: :attr:`numpy.ndarray.size`

        """

        return self._dparray_size

    @property
    def __array_interface__(self):
        # print(f"====__array_interface__====self._dparray_data={ < size_t > self._dparray_data}")
        interface_dict = {
            "data": ( < size_t > self._dparray_data, False),  # last parameter is "Writable"
            "strides": self.strides,
            "descr": None,
            "typestr": self.dtype.str,
            "shape": self.shape,
            "version": 3
        }

        return interface_dict


    @property
    def __sycl_usm_array_interface__(self):
        iface = {
            "data": (<Py_ssize_t>(<void *>self._dparray_data), False),
            "shape": self.shape,
            "strides": self.strides,
            "typestr": (lambda dt: dt.byteorder + dt.kind + str(dt.itemsize))(self._dparray_dtype),
            "version": 1,
            "syclobj": self.queue
        }
        return iface


    @property
    def _queue(self):
        return self.queue

    
    def __iter__(self):
        self.iter_idx = 0

        return self
        # if self.size == 0:
        # return # raise TypeError('__iter__ over a 0-d array')

        # Cython BUG Shows: "Unused entry 'genexpr'" if ""warn.unused": True"
        # https://github.com/cython/cython/issues/1699
        # return (self[i] for i in range(self._dparray_shape[0]))

    def __next__(self):
        cdef size_t prefix_idx = 0

        if self.iter_idx < self.size:
            prefix_idx = self.iter_idx
            self.iter_idx += 1
            return self[prefix_idx]

        raise StopIteration

    def __len__(self):
        """Returns the size of the first dimension.
        Equivalent to shape[0] and also equal to size only for one-dimensional arrays.

        .. seealso:: :attr:`numpy.ndarray.__len__`

        """

        if self.ndim == 0:
            raise TypeError('len() of an object with empty shape')

        return self._dparray_shape[0]

    def __getitem__(self, key):
        """Get the array item(s)
        x.__getitem__(key) <==> x[key]

        """

        if isinstance(key, tuple):
            """
            This is corner case for a[numpy.newaxis, ...] slicing
            didn't find good documentation about algorithm how to handle both of macros
            """
            if key == (None, Ellipsis):  # "key is (None, Ellipsis)" doesn't work
                result = dparray((1,) + self.shape, dtype=self.dtype)
                for i in range(result.size):
                    result[i] = self[i]

                return result

        lin_idx = utils._get_linear_index(key, self.shape, self.ndim)

        if lin_idx >= self.size:
            raise utils.checker_throw_index_error("__getitem__", lin_idx, self.size)

        if self.dtype == numpy.float64:
            return ( < double * > self._dparray_data)[lin_idx]
        elif self.dtype == numpy.float32:
            return ( < float * > self._dparray_data)[lin_idx]
        elif self.dtype == numpy.int64:
            return ( < long * > self._dparray_data)[lin_idx]
        elif self.dtype == numpy.int32:
            return ( < int * > self._dparray_data)[lin_idx]
        elif self.dtype == numpy.bool:
            return ( < cpp_bool * > self._dparray_data)[lin_idx]

        utils.checker_throw_type_error("__getitem__", self.dtype)

    def _setitem_scalar(self, key, value):
        """
        Set the array item by scalar value

        self[i] = 0.5
        """

        lin_idx = utils._get_linear_index(key, self.shape, self.ndim)

        if lin_idx >= self.size:
            raise utils.checker_throw_index_error("__setitem__", lin_idx, self.size)

        if self.dtype == numpy.float64:
            ( < double * > self._dparray_data)[lin_idx] = <double > value
        elif self.dtype == numpy.float32:
            ( < float * > self._dparray_data)[lin_idx] = <float > value
        elif self.dtype == numpy.int64:
            ( < long * > self._dparray_data)[lin_idx] = <long > value
        elif self.dtype == numpy.int32:
            ( < int * > self._dparray_data)[lin_idx] = <int > value
        elif self.dtype == numpy.bool:
            ( < cpp_bool * > self._dparray_data)[lin_idx] = < cpp_bool > value
        else:
            utils.checker_throw_type_error("__setitem__", self.dtype)

    def __setitem__(self, key, value):
        """Set the array item(s)
        x.__setitem__(key, value) <==> x[key] = value

        """
        if isinstance(key, slice):
            start = 0 if (key.start is None) else key.start
            stop = self.size if (key.stop is None) else key.stop
            step = 1 if (key.step is None) else key.step
            for i in range(start, stop, step):
                self._setitem_scalar(i, value[i])
        else:
            self._setitem_scalar(key, value)

    # -------------------------------------------------------------------------
    # Shape manipulation
    # -------------------------------------------------------------------------

    @property
    def flat(self):
        """ Return a flat iterator, or set a flattened version of self to value. """
        return self

    def flatten(self, order='C'):
        """
        Return a copy of the array collapsed into one dimension.

        Parameters
        ----------
        order: {'C', 'F', 'A', 'K'}, optional
            'C' means to flatten in row-major (C-style) order.
            'F' means to flatten in column-major (Fortran- style) order.
            'A' means to flatten in column-major order if a is Fortran contiguous in memory, row-major order otherwise.
            'K' means to flatten a in the order the elements occur in memory. The default is 'C'.

        Returns
        -------
        out: ndarray
            A copy of the input array, flattened to one dimension.

        See Also
        --------
        :obj:`dpnp.ravel`, :obj:`dpnp.flat`

        """

        if not utils.use_origin_backend(self):
            c_order, fortran_order = self.flags.c_contiguous, self.flags.f_contiguous

            if order not in {'C', 'F', 'A', 'K'}:
                pass
            elif order == 'K' and not c_order and not fortran_order:
                # skip dpnp backend if both C-style and Fortran-style order not found in flags
                pass
            else:
                if order == 'K':
                    # either C-style or Fortran-style found in flags
                    order = 'C' if c_order else 'F'
                elif order == 'A':
                    order = 'F' if fortran_order else 'C'

                if order == 'F':
                    return self.transpose().reshape(self.size)

                result = dparray(self.size, dtype=self.dtype)
                for i in range(result.size):
                    result[i] = self[i]

                return result

        result = utils.dp2nd_array(self).flatten(order=order)

        return utils.nd2dp_array(result)

    def ravel(self, order='C'):
        """
        Return a contiguous flattened array.

        Parameters
        ----------
        order: {'C', 'F', 'A', 'K'}, optional
            'C' means to flatten in row-major (C-style) order.
            'F' means to flatten in column-major (Fortran- style) order.
            'A' means to flatten in column-major order if a is Fortran contiguous in memory, row-major order otherwise.
            'K' means to flatten a in the order the elements occur in memory. The default is 'C'.

        Returns
        -------
        out: ndarray
            A copy of the input array, flattened to one dimension.

        See Also
        --------
        :obj:`dpnp.ravel`, :obj:`dpnp.flat`

        """
        # TODO: don't copy the input array
        return self.flatten(order=order)

    def reshape(self, shape, order=b'C'):
        """Change the shape of the array.

        .. seealso::
           :meth:`numpy.ndarray.reshape`

        """

        if order is not b'C':
            utils.checker_throw_value_error("dparray::reshape", "order", order, b'C')

        cdef long shape_it = 0
        cdef tuple shape_tup = utils._object_to_tuple(shape)
        cdef size_previous = self.size

        cdef long size_new = 1
        cdef dparray_shape_type shape_new
        shape_new.reserve(len(shape_tup))

        for shape_it in shape_tup:
            if shape_it < 0:
                utils.checker_throw_value_error("dparray::reshape", "shape", shape_it, ">=0")

            shape_new.push_back(shape_it)
            size_new *= shape_it

        if size_new != size_previous:
            utils.checker_throw_value_error("dparray::reshape", "shape", size_new, size_previous)

        self._dparray_shape = shape_new
        self._dparray_size = size_new

        return self

    def repeat(self, *args, **kwds):
        """ Repeat elements of an array.

        .. seealso::
           :obj:`dpnp.repeat` for full documentation,
           :meth:`numpy.ndarray.repeat`

        """
        return repeat(self, *args, **kwds)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """ Returns the variance of the array elements, along given axis.

        .. seealso::
           :obj:`dpnp.var` for full documentation,

        """
        return std(self, axis, dtype, out, ddof, keepdims)

    def transpose(self, *axes):
        """ Returns a view of the array with axes permuted.

        .. seealso::
           :obj:`dpnp.transpose` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        return transpose(self, axes)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """
        Returns the variance of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.var` for full documentation.

        See Also
        --------
        :obj:`numpy.ndarray.var` : corresponding function for ndarrays
        :obj:`numpy.var` : Equivalent function
        """
        return var(self, axis, dtype, out, ddof, keepdims)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ Return our function pointer regusted by `ufunc` parameter
        """

        if method == '__call__':
            name = ufunc.__name__

            import dpnp  # TODO need to remove this line

            try:
                dpnp_ufunc = getattr(dpnp, name)
            except AttributeError:
                utils.checker_throw_value_error("__array_ufunc__", "AttributeError in method", method, ufunc)

            return dpnp_ufunc(*inputs, **kwargs)
        else:
            print("DParray::__array_ufunc__ called")
            print("Arguments:")
            print(f"  ufunc={ufunc}, type={type(ufunc)}")
            print(f"  method={method}, type={type(method)}")
            for arg in inputs:
                print("  arg: ", arg)
            for key, value in kwargs.items():
                print("  kwargs: %s == %s" % (key, value))

            utils.checker_throw_value_error("__array_ufunc__", "method", method, ufunc)

    def __array_function__(self, func, types, args, kwargs):
        # print("\nDParray __array_function__ called")
        # print("Arguments:")
        # print(f"  func={func}, type={type(func)} from={func.__module__}")
        # print(f"  types={types}, type={type(types)}")
        # for arg in args:
        #     print("  arg: ", arg)
        # for key, value in kwargs.items():
        #     print("  kwargs: %s == %s" % (key, value))

        import dpnp  # TODO need to remove this line

        try:
            dpnp_func = getattr(dpnp, func.__name__)
        except AttributeError:
            utils.checker_throw_value_error("__array_function__", "AttributeError in method", types, func)

        return dpnp_func(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Comparison operations
    # -------------------------------------------------------------------------
    def __richcmp__(self, other, int op):
        if op == 0:
            return less(self, other)
        if op == 1:
            return less_equal(self, other)
        if op == 2:
            return equal(self, other)
        if op == 3:
            return not_equal(self, other)
        if op == 4:
            return greater(self, other)
        if op == 5:
            return greater_equal(self, other)

        utils.checker_throw_value_error("__richcmp__", "op", op, "0 <= op <=5")

    """
    -------------------------------------------------------------------------
    Unary operations
    -------------------------------------------------------------------------
    """

    def __matmul__(self, other):
        return matmul(self, other)

    """
    -------------------------------------------------------------------------
    Arithmetic operations
    -------------------------------------------------------------------------
    """

    # TODO: add scalar support
    def __add__(self, other):
        return add(self, other)

    def __mod__(self, other):
        return remainder(self, other)

    # TODO: add scalar support
    def __mul__(self, other):
        return multiply(self, other)

    def __neg__(self):
        return negative(self)

    # TODO: add scalar support
    def __pow__(self, other, modulo):
        return power(self, other, modulo=modulo)

    # TODO: add scalar support
    def __sub__(self, other):
        return subtract(self, other)

    # TODO: add scalar support
    def __truediv__(self, other):
        return divide(self, other)

    cpdef dparray astype(self, dtype, order='K', casting=None, subok=None, copy=True):
        """Copy the array with data type casting.

        Args:
            dtype: Target type.
            order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major (Fortran-style) order.
                When ``order`` is 'A', it uses 'F' if ``a`` is column-major and uses 'C' otherwise.
                And when ``order`` is 'K', it keeps strides as closely as possible.
            copy (bool): If it is False and no cast happens, then this method returns the array itself.
                Otherwise, a copy is returned.

        Returns:
            If ``copy`` is False and no cast is required, then the array itself is returned.
            Otherwise, it returns a (possibly casted) copy of the array.

        .. note::
           This method currently does not support `order``, `casting``, ``copy``, and ``subok`` arguments.

        .. seealso:: :meth:`numpy.ndarray.astype`

        """

        if casting is not None:
            utils.checker_throw_value_error("astype", "casting", casting, None)

        if subok is not None:
            utils.checker_throw_value_error("astype", "subok", subok, None)

        if copy is not True:
            utils.checker_throw_value_error("astype", "copy", copy, True)

        if order is not 'K':
            utils.checker_throw_value_error("astype", "order", order, 'K')

        return dpnp_astype(self, dtype)

    """
    -------------------------------------------------------------------------
    Calculation
    -------------------------------------------------------------------------
    """

    def sum(*args, **kwargs):
        """
        Returns the sum along a given axis.

        .. seealso::
           :obj:`dpnp.sum` for full documentation,
           :meth:`dpnp.dparray.sum`

        """

        # TODO don't know how to call `sum from python public interface. Simple call executes internal `sum` function.
        # numpy with dparray call public dpnp.sum via __array_interface__`
        return numpy.sum(*args, **kwargs)

    def max(self, axis=None):
        """
        Return the maximum along an axis.
        """

        return max(self, axis)

    def mean(self, axis=None):
        """
        Returns the average of the array elements.
        """

        return mean(self, axis)

    def min(self, axis=None):
        """
        Return the minimum along a given axis.
        """

        return min(self, axis)

    """
    -------------------------------------------------------------------------
    Sorting
    -------------------------------------------------------------------------
    """

    def argsort(self, axis=-1, kind=None, order=None):
        """
        Return an ndarray of indices that sort the array along the
        specified axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. If None, the default, the flattened array
            is used.
            ..  versionchanged:: 1.13.0
                Previously, the default was documented to be -1, but that was
                in error. At some future date, the default will change to -1, as
                originally intended.
                Until then, the axis should be given explicitly when
                ``arr.ndim > 1``, to avoid a FutureWarning.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            The sorting algorithm used.
        order : list, optional
            When `a` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  Not all fields need be
            specified.

        Returns
        -------
        index_array : ndarray, int
            Array of indices that sort `a` along the specified axis.
            In other words, ``a[index_array]`` yields a sorted `a`.

        See Also
        --------
        MaskedArray.sort : Describes sorting algorithms used.
        :obj:`dpnp.lexsort` : Indirect stable sort with multiple keys.
        :obj:`numpy.ndarray.sort` : Inplace sort.

        Notes
        -----
        See `sort` for notes on the different sorting algorithms.

        """
        return argsort(self, axis, kind, order)

    def sort(self, axis=-1, kind=None, order=None):
        """
        Sort the array

        Parameters
        ----------
        a : array_like
            Array to be sorted.
        axis : int, optional
            Axis along which to sort. If None, the array is flattened before
            sorting. The default is -1, which sorts along the last axis.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            The sorting algorithm used.
        order : list, optional
            When `a` is a structured array, this argument specifies which fields
            to compare first, second, and so on.  This list does not need to
            include all of the fields.

        Returns
        -------
        sorted_array : ndarray
            Array of the same type and shape as `a`.

        See Also
        --------
        :obj:`numpy.ndarray.sort` : Method to sort an array in-place.
        :obj:`dpnp.argsort` : Indirect sort.
        :obj:`dpnp.lexsort` : Indirect stable sort on multiple keys.
        :obj:`dpnp.searchsorted` : Find elements in a sorted array.

        Notes
        -----
        See ``sort`` for notes on the different sorting algorithms.

        """
        return sort(self, axis, kind, order)

    """
    -------------------------------------------------------------------------
    Searching
    -------------------------------------------------------------------------
    """

    def argmax(self, axis=None, out=None):
        """
        Returns array of indices of the maximum values along the given axis.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        index_array : {integer_array}

        Examples
        --------
        >>> a = np.arange(6).reshape(2,3)
        >>> a.argmax()
        5
        >>> a.argmax(0)
        array([1, 1, 1])
        >>> a.argmax(1)
        array([2, 2])

        """
        return argmax(self, axis, out)

    def argmin(self, axis=None, out=None):
        """
        Return array of indices to the minimum values along the given axis.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        ndarray or scalar
            If multi-dimension input, returns a new ndarray of indices to the
            minimum values along the given axis.  Otherwise, returns a scalar
            of index to the minimum values along the given axis.

        """
        return argmin(self, axis, out)

    """
    -------------------------------------------------------------------------
    Logic
    -------------------------------------------------------------------------
    """

    def all(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.

        Refer to `numpy.all` for full documentation.

        See Also
        --------
        :obj:`numpy.all` : equivalent function

        """

        return all(self, axis, out, keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        """
        Returns True if any of the elements of `a` evaluate to True.

        Refer to `numpy.any` for full documentation.

        See Also
        --------
        :obj:`numpy.any` : equivalent function

        """

        return any(self, axis, out, keepdims)

    """
    -------------------------------------------------------------------------
    Other attributes
    -------------------------------------------------------------------------
    """

    @property
    def T(self):
        """Shape-reversed view of the array.

        If ndim < 2, then this is just a reference to the array itself.

        """
        if self.ndim < 2:
            return self
        else:
            return transpose(self)

    cpdef item(self, size_t id):
        """Return the item described by id."""
        return self[id]

    cdef void * get_data(self):
        return self._dparray_data

    def fill(self, value):
        """
        Fill the array with a scalar value.

        Parameters
        ----------
        value : scalar
            All elements of `a` will be assigned this value.

        Examples
        --------
        >>> a = np.array([1, 2])
        >>> a.fill(0)
        >>> a
        array([0, 0])
        >>> a = np.empty(2)
        >>> a.fill(1)
        >>> a
        array([1.,  1.])
        """

        for i in range(self.size):
            self[i] = value
