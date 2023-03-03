# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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

import dpctl.tensor as dpt
import numpy

import dpnp

class dpnp_array:
    """
    Multi-dimensional array object.

    This is a wrapper around dpctl.tensor.usm_ndarray that provides
    methods to be compliant with original Numpy.

    """

    def __init__(self,
                 shape,
                 dtype="f8",
                 buffer=None,
                 offset=0,
                 strides=None,
                 order="C",
                 device=None,
                 usm_type="device",
                 sycl_queue=None):
        if buffer is not None:
            if not isinstance(buffer, dpt.usm_ndarray):
                raise TypeError(
                    "Expected dpctl.tensor.usm_ndarray, got {}"
                    "".format(type(buffer))
                )
            if buffer.shape != shape:
                raise ValueError(
                    "Expected buffer.shape={}, got {}"
                    "".format(shape, buffer.shape)
                )
            self._array_obj = dpt.asarray(buffer,
                                          copy=False,
                                          order=order)
        else:
            sycl_queue_normalized = dpnp.get_normalized_queue_device(device=device, sycl_queue=sycl_queue)
            self._array_obj = dpt.usm_ndarray(shape,
                                              dtype=dtype,
                                              strides=strides,
                                              buffer=usm_type,
                                              offset=offset,
                                              order=order,
                                              buffer_ctor_kwargs={"queue": sycl_queue_normalized})

    @property
    def __sycl_usm_array_interface__(self):
        return self._array_obj.__sycl_usm_array_interface__

    def get_array(self):
        """Get usm_ndarray object."""
        return self._array_obj

    @property
    def T(self):
        """Shape-reversed view of the array.

        If ndim < 2, then this is just a reference to the array itself.

        """
        if self.ndim < 2:
            return self
        else:
            return dpnp.transpose(self)

    def to_device(self, target_device):
        """
        Transfer array to target device
        """

        return dpnp_array(shape=self.shape, buffer=self.get_array().to_device(target_device))

    @property
    def sycl_queue(self):
        return self._array_obj.sycl_queue

    @property
    def sycl_device(self):
        return self._array_obj.sycl_device

    @property
    def sycl_context(self):
        return self._array_obj.sycl_context

    @property
    def device(self):
        return self._array_obj.device

    @property
    def usm_type(self):
        return self._array_obj.usm_type

    def __abs__(self):
        return dpnp.abs(self)

    def __add__(self, other):
        return dpnp.add(self, other)

 # '__and__',
 # '__array__',
 # '__array_finalize__',
 # '__array_function__',
 # '__array_interface__',
 # '__array_prepare__',
 # '__array_priority__',
 # '__array_struct__',
 # '__array_ufunc__',
 # '__array_wrap__',

    def __bool__(self):
        return self._array_obj.__bool__()

 # '__class__',

    def __complex__(self):
        return self._array_obj.__complex__()

 # '__contains__',
 # '__copy__',
 # '__deepcopy__',
 # '__delattr__',
 # '__delitem__',
 # '__dir__',
 # '__divmod__',
 # '__doc__',

    def __dlpack__(self, stream=None):
        return self._array_obj.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self._array_obj.__dlpack_device__()

    def __eq__(self, other):
        return dpnp.equal(self, other)

    def __float__(self):
        return self._array_obj.__float__()

 # '__floordiv__',
 # '__format__',

    def __ge__(self, other):
        return dpnp.greater_equal(self, other)

 # '__getattribute__',

    def __getitem__(self, key):
        if isinstance(key, dpnp_array):
            key = key.get_array()

        item = self._array_obj.__getitem__(key)
        if not isinstance(item, dpt.usm_ndarray):
            raise RuntimeError(
                "Expected dpctl.tensor.usm_ndarray, got {}"
                "".format(type(item)))

        res = self.__new__(dpnp_array)
        res._array_obj = item

        return res

    def __gt__(self, other):
        return dpnp.greater(self, other)

 # '__hash__',
 # '__iadd__',
 # '__iand__',
 # '__ifloordiv__',
 # '__ilshift__',
 # '__imatmul__',
 # '__imod__',
 # '__imul__',

    def __index__(self):
        return self._array_obj.__index__()

 # '__init__',
 # '__init_subclass__',

    def __int__(self):
        return self._array_obj.__int__()

 # '__invert__',
 # '__ior__',

    def __ipow__(self, other):
        dpnp.power(self, other, out=self)
        return self

 # '__irshift__',
 # '__isub__',
 # '__iter__',
 # '__itruediv__',
 # '__ixor__',

    def __le__(self, other):
        return dpnp.less_equal(self, other)

    def __len__(self):
        """
        Performs the operation __len__.
        """

        return self._array_obj.__len__()

 # '__lshift__',

    def __lt__(self, other):
        return dpnp.less(self, other)

    def __matmul__(self, other):
        return dpnp.matmul(self, other)

    def __mod__(self, other):
        return dpnp.remainder(self, other)

    def __mul__(self, other):
        return dpnp.multiply(self, other)

    def __ne__(self, other):
        return dpnp.not_equal(self, other)

    def __neg__(self):
        return dpnp.negative(self)

 # '__new__',
 # '__or__',
 # '__pos__',

    def __pow__(self, other):
        return dpnp.power(self, other)

    def __radd__(self, other):
        return dpnp.add(other, self)

 # '__rand__',
 # '__rdivmod__',
 # '__reduce__',
 # '__reduce_ex__',

    def __repr__(self):
        return dpt.usm_ndarray_repr(self._array_obj, prefix="array")

 # '__rfloordiv__',
 # '__rlshift__',

    def __rmatmul__(self, other):
        return dpnp.matmul(other, self)

    def __rmod__(self, other):
        return remainder(other, self)

    def __rmul__(self, other):
        return dpnp.multiply(other, self)

 # '__ror__',
 
    def __rpow__(self, other):
        return dpnp.power(other, self)

 # '__rrshift__',
 # '__rshift__',

    def __rsub__(self, other):
        return dpnp.subtract(other, self)

    def __rtruediv__(self, other):
        return dpnp.true_divide(other, self)

 # '__rxor__',
 # '__setattr__',

    def __setitem__(self, key, val):
        if isinstance(key, dpnp_array):
            key = key.get_array()
        if isinstance(val, dpnp_array):
            val = val.get_array()

        self._array_obj.__setitem__(key, val)

 # '__setstate__',
 # '__sizeof__',


    def __str__(self):
        """
        Output values from the array to standard output.

        Examples
        --------
        >>> print(a)
        [[ 136.  136.  136.]
         [ 272.  272.  272.]
         [ 408.  408.  408.]]

        """

        return self._array_obj.__str__()

    def __sub__(self, other):
        return dpnp.subtract(self, other)

 # '__subclasshook__',

    def __truediv__(self, other):
        return dpnp.true_divide(self, other)

 # '__xor__',

    @staticmethod
    def _create_from_usm_ndarray(usm_ary : dpt.usm_ndarray):
        if not isinstance(usm_ary, dpt.usm_ndarray):
            raise TypeError(
                f"Expected dpctl.tensor.usm_ndarray, got {type(usm_ary)}"
                )
        res = dpnp_array.__new__(dpnp_array)
        res._array_obj = usm_ary
        return res

    def all(self,
            axis=None,
            out=None,
            keepdims=False,
            *,
            where=True):
        """
        Returns True if all elements evaluate to True.

        Refer to :obj:`dpnp.all` for full documentation.

        See Also
        --------
        :obj:`dpnp.all` : equivalent function

        """

        return dpnp.all(self, axis=axis, out=out, keepdims=keepdims, where=where)

    def any(self,
            axis=None,
            out=None,
            keepdims=False,
            *,
            where=True):
        """
        Returns True if any of the elements of `a` evaluate to True.

        Refer to :obj:`dpnp.any` for full documentation.

        See Also
        --------
        :obj:`dpnp.any` : equivalent function

        """

        return dpnp.any(self, axis=axis, out=out, keepdims=keepdims, where=where)

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
        return dpnp.argmax(self, axis, out)

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
        return dpnp.argmin(self, axis, out)

# 'argpartition',

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
        return dpnp.argsort(self, axis, kind, order)


    def asnumpy(self):
        """
        Copy content of the array into :class:`numpy.ndarray` instance of the same shape and data type.

        Returns
        -------
        numpy.ndarray
            An instance of :class:`numpy.ndarray` populated with the array content.

        """

        return dpt.asnumpy(self._array_obj)


    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
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

        new_array = self.__new__(dpnp_array)
        new_array._array_obj = dpt.astype(self._array_obj, dtype, order=order, casting=casting, copy=copy)
        return new_array

 # 'base',
 # 'byteswap',

    def choose(input, choices, out=None, mode='raise'):
        """
        Construct an array from an index array and a set of arrays to choose from.

        """

        return dpnp.choose(input, choices, out, mode)

 # 'clip',
 # 'compress',

    def conj(self):
        """
        Complex-conjugate all elements.

        For full documentation refer to :obj:`numpy.ndarray.conj`.

        """

        if not dpnp.issubsctype(self.dtype, dpnp.complex_):
            return self
        else:
            return dpnp.conjugate(self)

    def conjugate(self):
        """
        Return the complex conjugate, element-wise.

        For full documentation refer to :obj:`numpy.ndarray.conjugate`.

        """

        if not dpnp.issubsctype(self.dtype, dpnp.complex_):
            return self
        else:
            return dpnp.conjugate(self)

 # 'copy',
 # 'ctypes',
 # 'cumprod',

    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Return the cumulative sum of the elements along the given axis.

        See Also
        --------
        :obj:`dpnp.cumsum`

        """

        return dpnp.cumsum(self, axis=axis, dtype=dtype, out=out)

 # 'data',

    def diagonal(input, offset=0, axis1=0, axis2=1):
        """
        Return specified diagonals.

        See Also
        --------
        :obj:`dpnp.diagonal`

        """

        return dpnp.diagonal(input, offset, axis1, axis2)

 # 'dot',

    @property
    def dtype(self):
        """
        """

        return self._array_obj.dtype

 # 'dump',
 # 'dumps',

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
            self.flat[i] = value

    @property
    def flags(self):
        """
        Return information about the memory layout of the array.

        """

        return self._array_obj.flags

    @property
    def flat(self):
        """
        Return a flat iterator, or set a flattened version of self to value.

        """

        return dpnp.flatiter(self)

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
        new_arr = self.__new__(dpnp_array)
        new_arr._array_obj = dpt.empty(self.shape,
                                       dtype=self.dtype,
                                       order=order,
                                       device=self._array_obj.sycl_device,
                                       usm_type=self._array_obj.usm_type,
                                       sycl_queue=self._array_obj.sycl_queue)

        if self.size > 0:
            dpt._copy_utils._copy_from_usm_ndarray_to_usm_ndarray(new_arr._array_obj, self._array_obj)
            new_arr._array_obj = dpt.reshape(new_arr._array_obj, (self.size, ))

        return new_arr

 # 'getfield',
 # 'imag',

    def item(self, id=None):
        """
        Copy an element of an array to a standard Python scalar and return it.

        For full documentation refer to :obj:`numpy.ndarray.item`.

        Examples
        --------
        >>> np.random.seed(123)
        >>> x = np.random.randint(9, size=(3, 3))
        >>> x
        array([[2, 2, 6],
               [1, 3, 6],
               [1, 0, 1]])
        >>> x.item(3)
        1
        >>> x.item(7)
        0
        >>> x.item((0, 1))
        2
        >>> x.item((2, 2))
        1

        """

        if id is None:
            if self.size != 1:
                raise ValueError("DPNP dparray::item(): can only convert an array of size 1 to a Python scalar")
            else:
                id = 0

        return self.flat[id]

 # 'itemset',

    @property
    def itemsize(self):
        """
        """

        return self._array_obj.itemsize

    def max(self, axis=None, out=None, keepdims=numpy._NoValue, initial=numpy._NoValue, where=numpy._NoValue):
        """
        Return the maximum along an axis.
        """

        return dpnp.max(self, axis, out, keepdims, initial, where)

    def mean(self, axis=None):
        """
        Returns the average of the array elements.
        """

        return dpnp.mean(self, axis)

    def min(self, axis=None, out=None, keepdims=numpy._NoValue, initial=numpy._NoValue, where=numpy._NoValue):
        """
        Return the minimum along a given axis.
        """

        return dpnp.min(self, axis, out, keepdims, initial, where)

 # 'nbytes',

    @property
    def ndim(self):
        """
        """

        return self._array_obj.ndim

 # 'newbyteorder',
 # 'nonzero',
 # 'partition',

    def prod(self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
        """
        Returns the prod along a given axis.

        .. seealso::
           :obj:`dpnp.prod` for full documentation,
           :meth:`dpnp.dparray.sum`

        """

        return dpnp.prod(self, axis, dtype, out, keepdims, initial, where)

 # 'ptp',
 # 'put',
 # 'ravel',
 # 'real',
 # 'repeat',

    def reshape(self, d0, *dn, order=b'C'):
        """
        Returns an array containing the same data with a new shape.

        Refer to `dpnp.reshape` for full documentation.

        .. seealso::
           :meth:`numpy.ndarray.reshape`

        Notes
        -----
        Unlike the free function `dpnp.reshape`, this method on `ndarray` allows
        the elements of the shape parameter to be passed in as separate arguments.
        For example, ``a.reshape(10, 11)`` is equivalent to
        ``a.reshape((10, 11))``.

        """

        if dn:
            if not isinstance(d0, int):
                msg_tmpl = "'{}' object cannot be interpreted as an integer"
                raise TypeError(msg_tmpl.format(type(d0).__name__))
            shape = [d0, *dn]
        else:
            shape = d0

        shape_tup = dpnp.dpnp_utils._object_to_tuple(shape)

        return dpnp.reshape(self, shape_tup)

 # 'resize',

    def round(self, decimals=0, out=None):
        """
        Return array with each element rounded to the given number of decimals.

        .. seealso::
           :obj:`dpnp.around` for full documentation.

        """

        return dpnp.around(self, decimals, out)

 # 'searchsorted',
 # 'setfield',
 # 'setflags',

    @property
    def shape(self):
        """Lengths of axes. A tuple of numbers represents size of each dimention.

        Setter of this property involves reshaping without copy. If the array
        cannot be reshaped without copy, it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """

        return self._array_obj.shape

    @shape.setter
    def shape(self, newshape):
        """Set new lengths of axes. A tuple of numbers represents size of each dimention.
        It involves reshaping without copy. If the array cannot be reshaped without copy,
        it raises an exception.

        .. seealso: :attr:`numpy.ndarray.shape`

        """

        dpnp.reshape(self, newshape)

    @property
    def shape(self):
        """
        """

        return self._array_obj.shape

    @property
    def size(self):
        """
        """

        return self._array_obj.size

 # 'sort',

    def squeeze(self, axis=None):
        """
        Remove single-dimensional entries from the shape of an array.

        .. seealso::
           :obj:`dpnp.squeeze` for full documentation

        """

        return dpnp.squeeze(self, axis)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """ Returns the variance of the array elements, along given axis.

        .. seealso::
           :obj:`dpnp.var` for full documentation,

        """

        return dpnp.std(self, axis, dtype, out, ddof, keepdims)

    @property
    def strides(self):
        """
        """

        return self._array_obj.strides

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
        """
        Returns the sum along a given axis.

        .. seealso::
           :obj:`dpnp.sum` for full documentation,
           :meth:`dpnp.dparray.sum`

        """

        return dpnp.sum(self, axis, dtype, out, keepdims, initial, where)

 # 'swapaxes',

    def take(self, indices, axis=None, out=None, mode='raise'):
        """
        Take elements from an array.

        For full documentation refer to :obj:`numpy.take`.

        """

        return dpnp.take(self, indices, axis, out, mode)

 # 'tobytes',
 # 'tofile',
 # 'tolist',
 # 'tostring',
 # 'trace',

    def transpose(self, *axes):
        """
        Returns a view of the array with axes permuted.

        .. seealso::
           :obj:`dpnp.transpose` for full documentation,
           :meth:`numpy.ndarray.reshape`

        """

        return dpnp.transpose(self, axes)

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

        return dpnp.var(self, axis, dtype, out, ddof, keepdims)

 # 'view'
