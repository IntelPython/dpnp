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
from builtins import None

"""Module DPArray

This module contains Array class represents multi-dimensional array
using USB interface for an Intel GPU device.

"""


from libcpp cimport bool

from dpnp.dpnp_iface_types import *
from dpnp.dpnp_iface import *
from dpnp.backend cimport *
import numpy
cimport numpy

cimport dpnp.dpnp_utils as utils


cdef class dparray:
    """Multi-dimensional array using USM interface for an Intel GPU device.

    This class implements a subset of methods of :class:`numpy.ndarray`.
    The difference is that this class allocates the array content useing
    USM interface on the current GPU device.

    Args:
        shape (tuple of ints): Length of axes.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        memptr (char *): Pointer to the array content head.
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

    def __init__(self, shape, dtype=float64, memptr=None, strides=None, order=b'C'):
        cdef Py_ssize_t shape_it = 0
        cdef Py_ssize_t strides_it = 0

        if order != b'C':
            order = utils._normalize_order(order)

        if order != b'C' and order != b'F':
            raise TypeError("Intel NumPy dparray::__init__(): Parameter is not understood. order={}".format(order))

        # dtype
        self._dparray_dtype = numpy.dtype(dtype)

        # size and shape
        cdef tuple shape_tup = utils._object_to_tuple(shape)
        self._dparray_shape.reserve(len(shape_tup))

        self._dparray_size = 1
        for shape_it in shape_tup:
            if shape_it < 0:
                raise ValueError("Intel NumPy dparray::__init__(): Negative values in 'shape' are not allowed")
            # shape
            self._dparray_shape.push_back(shape_it)
            # size
            self._dparray_size *= shape_it

        # strides
        cdef tuple strides_tup = utils._object_to_tuple(strides)
        self._dparray_strides.reserve(len(strides_tup))
        for strides_it in strides_tup:
            if strides_it < 0:
                raise ValueError("Intel NumPy dparray::__init__(): Negative values in 'strides' are not allowed")
            # stride
            self._dparray_strides.push_back(strides_it)

        # data
        if memptr is None:
            self._dparray_data = dpnp_memory_alloc_c(self.nbytes)
        else:
            self._dparray_data = memptr

    def __dealloc__(self):
        """ Release owned memory

        """

        dpnp_memory_free_c(self._dparray_data)

    def __repr__(self):
        """ Output information about the array to standard output

        Example:
          <Intel Numpy DParray:name=dparray: mem=0x7ffad6fa4000: size=1048576: shape=[1024, 1024]: type=float64>

        """

        string = "<Intel Numpy DParray:name={}".format(self.__class__.__name__)
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
    def flags(self):
        """Object containing memory-layout information.

        It only contains ``c_contiguous``, ``f_contiguous``, and ``owndata`` attributes.
        All of these are read-only. Accessing by indexes is also supported.

        .. seealso:: :attr:`numpy.ndarray.flags`

        """

        return (True, False, False)

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
            return ( < bool * > self._dparray_data)[lin_idx]

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
            ( < bool * > self._dparray_data)[lin_idx] = <bool > value
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
           :func:`dpnp.repeat` for full documentation,
           :meth:`numpy.ndarray.repeat`

        """
        return repeat(self, *args, **kwds)

    def transpose(self, *axes):
        """ Returns a view of the array with axes permuted.

        .. seealso::
           :func:`dpnp.transpose` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """
        return transpose(self, axes)

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
           :func:`dpnp.sum` for full documentation,
           :meth:`dpnp.dparray.sum`

        """

        # TODO don't know how to call `sum from python public interface. Simple call executes internal `sum` function.
        # numpy with dparray call public dpnp.sum via __array_interface__`
        return numpy.sum(*args, **kwargs)

    def mean(self, axis=None):
        """
        Returns the average of the array elements.
        """

        return mean(self, axis)

    """
    -------------------------------------------------------------------------
    Sorting
    -------------------------------------------------------------------------
    """

    def argsort(self, axis=-1, kind=None, order=None):
        """
        Return an ndarray of indices that sort the array along the
        specified axis.  Masked values are filled beforehand to
        `fill_value`.
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
        endwith : {True, False}, optional
            Whether missing values (if any) should be treated as the largest values
            (True) or the smallest values (False)
            When the array contains unmasked values at the same extremes of the
            datatype, the ordering of these values and the masked values is
            undefined.
        fill_value : {var}, optional
            Value used internally for the masked values.
            If ``fill_value`` is not None, it supersedes ``endwith``.
        Returns
        -------
        index_array : ndarray, int
            Array of indices that sort `a` along the specified axis.
            In other words, ``a[index_array]`` yields a sorted `a`.
        See Also
        --------
        MaskedArray.sort : Describes sorting algorithms used.
        lexsort : Indirect stable sort with multiple keys.
        numpy.ndarray.sort : Inplace sort.
        Notes
        -----
        See `sort` for notes on the different sorting algorithms.
        Examples
        --------
        >>> a = np.ma.array([3,2,1], mask=[False, False, True])
        >>> a
        masked_array(data=[3, 2, --],
                     mask=[False, False,  True],
               fill_value=999999)
        >>> a.argsort()
        array([1, 0, 2])

        """

        return argsort(self, axis, kind, order)

    def sort(self, axis=-1, kind=None, order=None):
        """
        Return a sorted copy of an array.
        Parameters
        ----------
        a : array_like
            Array to be sorted.
        axis : int or None, optional
            Axis along which to sort. If None, the array is flattened before
            sorting. The default is -1, which sorts along the last axis.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
            and 'mergesort' use timsort or radix sort under the covers and, in general,
            the actual implementation will vary with data type. The 'mergesort' option
            is retained for backwards compatibility.
            .. versionchanged:: 1.15.0.
               The 'stable' option was added.
        order : str or list of str, optional
            When `a` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  A single field can
            be specified as a string, and not all fields need be specified,
            but unspecified fields will still be used, in the order in which
            they come up in the dtype, to break ties.
        Returns
        -------
        sorted_array : ndarray
            Array of the same type and shape as `a`.
        See Also
        --------
        ndarray.sort : Method to sort an array in-place.
        argsort : Indirect sort.
        lexsort : Indirect stable sort on multiple keys.
        searchsorted : Find elements in a sorted array.
        partition : Partial sort.
        Notes
        -----
        The various sorting algorithms are characterized by their average speed,
        worst case performance, work space size, and whether they are stable. A
        stable sort keeps items with the same key in the same relative
        order. The four algorithms implemented in NumPy have the following
        properties:
        =========== ======= ============= ============ ========
           kind      speed   worst case    work space   stable
        =========== ======= ============= ============ ========
        'quicksort'    1     O(n^2)            0          no
        'heapsort'     3     O(n*log(n))       0          no
        'mergesort'    2     O(n*log(n))      ~n/2        yes
        'timsort'      2     O(n*log(n))      ~n/2        yes
        =========== ======= ============= ============ ========
        .. note:: The datatype determines which of 'mergesort' or 'timsort'
           is actually used, even if 'mergesort' is specified. User selection
           at a finer scale is not currently available.
        All the sort algorithms make temporary copies of the data when
        sorting along any but the last axis.  Consequently, sorting along
        the last axis is faster and uses less space than sorting along
        any other axis.
        The sort order for complex numbers is lexicographic. If both the real
        and imaginary parts are non-nan then the order is determined by the
        real parts except when they are equal, in which case the order is
        determined by the imaginary parts.
        Previous to numpy 1.4.0 sorting real and complex arrays containing nan
        values led to undefined behaviour. In numpy versions >= 1.4.0 nan
        values are sorted to the end. The extended sort order is:
          * Real: [R, nan]
          * Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]
        where R is a non-nan real value. Complex values with the same nan
        placements are sorted according to the non-nan part if it exists.
        Non-nan values are sorted as before.
        .. versionadded:: 1.12.0
        quicksort has been changed to `introsort <https://en.wikipedia.org/wiki/Introsort>`_.
        When sorting does not make enough progress it switches to
        `heapsort <https://en.wikipedia.org/wiki/Heapsort>`_.
        This implementation makes quicksort O(n*log(n)) in the worst case.
        'stable' automatically chooses the best stable sorting algorithm
        for the data type being sorted.
        It, along with 'mergesort' is currently mapped to
        `timsort <https://en.wikipedia.org/wiki/Timsort>`_
        or `radix sort <https://en.wikipedia.org/wiki/Radix_sort>`_
        depending on the data type.
        API forward compatibility currently limits the
        ability to select the implementation and it is hardwired for the different
        data types.
        .. versionadded:: 1.17.0
        Timsort is added for better performance on already or nearly
        sorted data. On random data timsort is almost identical to
        mergesort. It is now used for stable sort while quicksort is still the
        default sort if none is chosen. For timsort details, refer to
        `CPython listsort.txt <https://github.com/python/cpython/blob/3.7/Objects/listsort.txt>`_.
        'mergesort' and 'stable' are mapped to radix sort for integer data types. Radix sort is an
        O(n) sort instead of O(n log n).
        .. versionchanged:: 1.18.0
        NaT now sorts to the end of arrays for consistency with NaN.
        Examples
        --------
        >>> a = np.array([[1,4],[3,1]])
        >>> np.sort(a)                # sort along the last axis
        array([[1, 4],
               [1, 3]])
        >>> np.sort(a, axis=None)     # sort the flattened array
        array([1, 1, 3, 4])
        >>> np.sort(a, axis=0)        # sort along the first axis
        array([[1, 1],
               [3, 4]])
        Use the `order` keyword to specify a field to use when sorting a
        structured array:
        >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
        >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
        ...           ('Galahad', 1.7, 38)]
        >>> a = np.array(values, dtype=dtype)       # create a structured array
        >>> np.sort(a, order='height')                        # doctest: +SKIP
        array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
               ('Lancelot', 1.8999999999999999, 38)],
              dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
        Sort by age, then height if ages are equal:
        >>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP
        array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
               ('Arthur', 1.8, 41)],
              dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
        """

        return sort(self, axis, kind, order)

    """
    -------------------------------------------------------------------------
    Searching
    -------------------------------------------------------------------------
    """

    def argmax(self, axis=None, out=None):
        """
        Returns the indices of the maximum values along an axis.
        Parameters
        ----------
        a : array_like
            Input array.
        axis : int, optional
            By default, the index is into the flattened array, otherwise
            along the specified axis.
        out : array, optional
            If provided, the result will be inserted into this array. It should
            be of the appropriate shape and dtype.
        Returns
        -------
        index_array : ndarray of ints
            Array of indices into the array. It has the same shape as `a.shape`
            with the dimension along `axis` removed.
        See Also
        --------
        ndarray.argmax, argmin
        amax : The maximum value along a given axis.
        unravel_index : Convert a flat index into an index tuple.
        take_along_axis : Apply ``np.expand_dims(index_array, axis)``
                          from argmax to an array as if by calling max.
        Notes
        -----
        In case of multiple occurrences of the maximum values, the indices
        corresponding to the first occurrence are returned.
        Examples
        --------
        >>> a = np.arange(6).reshape(2,3) + 10
        >>> a
        array([[10, 11, 12],
               [13, 14, 15]])
        >>> np.argmax(a)
        5
        >>> np.argmax(a, axis=0)
        array([1, 1, 1])
        >>> np.argmax(a, axis=1)
        array([2, 2])
        Indexes of the maximal elements of a N-dimensional array:
        >>> ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
        >>> ind
        (1, 2)
        >>> a[ind]
        15
        >>> b = np.arange(6)
        >>> b[1] = 5
        >>> b
        array([0, 5, 2, 3, 4, 5])
        >>> np.argmax(b)  # Only the first occurrence is returned.
        1
        >>> x = np.array([[4,2,3], [1,0,3]])
        >>> index_array = np.argmax(x, axis=-1)
        >>> # Same as np.max(x, axis=-1, keepdims=True)
        >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
        array([[4],
               [3]])
        >>> # Same as np.max(x, axis=-1)
        >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1).squeeze(axis=-1)
        array([4, 3])
        """

        return argmax(self, axis, out)

    def argmin(self, axis=None, out=None):
        """
        Returns the indices of the minimum values along an axis.
        Parameters
        ----------
        a : array_like
            Input array.
        axis : int, optional
            By default, the index is into the flattened array, otherwise
            along the specified axis.
        out : array, optional
            If provided, the result will be inserted into this array. It should
            be of the appropriate shape and dtype.
        Returns
        -------
        index_array : ndarray of ints
            Array of indices into the array. It has the same shape as `a.shape`
            with the dimension along `axis` removed.
        See Also
        --------
        ndarray.argmin, argmax
        amin : The minimum value along a given axis.
        unravel_index : Convert a flat index into an index tuple.
        take_along_axis : Apply ``np.expand_dims(index_array, axis)``
                          from argmin to an array as if by calling min.
        Notes
        -----
        In case of multiple occurrences of the minimum values, the indices
        corresponding to the first occurrence are returned.
        Examples
        --------
        >>> a = np.arange(6).reshape(2,3) + 10
        >>> a
        array([[10, 11, 12],
               [13, 14, 15]])
        >>> np.argmin(a)
        0
        >>> np.argmin(a, axis=0)
        array([0, 0, 0])
        >>> np.argmin(a, axis=1)
        array([0, 0])
        Indices of the minimum elements of a N-dimensional array:
        >>> ind = np.unravel_index(np.argmin(a, axis=None), a.shape)
        >>> ind
        (0, 0)
        >>> a[ind]
        10
        >>> b = np.arange(6) + 10
        >>> b[4] = 10
        >>> b
        array([10, 11, 12, 13, 10, 15])
        >>> np.argmin(b)  # Only the first occurrence is returned.
        0
        >>> x = np.array([[4,2,3], [1,0,3]])
        >>> index_array = np.argmin(x, axis=-1)
        >>> # Same as np.min(x, axis=-1, keepdims=True)
        >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
        array([[2],
               [0]])
        >>> # Same as np.max(x, axis=-1)
        >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1).squeeze(axis=-1)
        array([2, 0])
        """

        return argmin(self, axis, out)

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
        return self[id]

    cdef void * get_data(self):
        return self._dparray_data
