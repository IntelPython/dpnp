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

import dpctl
import dpnp
import numpy

class dpnp_array:
    """
    Multi-dimensional array object.

    This is a wrapper around dpctl.tensor.usm_ndarray that provide
    methods to be complient with original Numpy.

    """

    def __init__(self, shape, dtype=numpy.float64):
        self._array_obj = dpctl.tensor.usm_ndarray(shape, dtype=dtype)

    @property
    def __sycl_usm_array_interface__(self):
        return self._array_obj.__sycl_usm_array_interface__

    @property
    def T(self):
        """Shape-reversed view of the array.

        If ndim < 2, then this is just a reference to the array itself.

        """
        if self.ndim < 2:
            return self
        else:
            return dpnp.transpose(self)

    def __abs__(self):
        return dpnp.abs(self)

 # '__add__',
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
 # '__complex__',
 # '__contains__',
 # '__copy__',
 # '__deepcopy__',
 # '__delattr__',
 # '__delitem__',
 # '__dir__',
 # '__divmod__',
 # '__doc__',

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
        return self._array_obj.__getitem__(key)

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
 # '__index__',
 # '__init__',
 # '__init_subclass__',

    def __int__(self):
        return self._array_obj.__int__()

 # '__invert__',
 # '__ior__',
 # '__ipow__',
 # '__irshift__',
 # '__isub__',
 # '__iter__',
 # '__itruediv__',
 # '__ixor__',

    def __le__(self, other):
        return dpnp.less_equal(self, other)

 # '__len__',
 # '__lshift__',

    def __lt__(self, other):
        return dpnp.less(self, other)

    def __matmul__(self, other):
        return dpnp.matmul(self, other)

 # '__mod__',

    def __mul__(self, other):
        return dpnp.multiply(self, other)

    def __ne__(self, other):
        return dpnp.not_equal(self, other)

 # '__neg__',
 # '__new__',
 # '__or__',
 # '__pos__',
 # '__pow__',
 # '__radd__',
 # '__rand__',
 # '__rdivmod__',
 # '__reduce__',
 # '__reduce_ex__',
 # '__repr__',
 # '__rfloordiv__',
 # '__rlshift__',

    def __rmatmul__(self, other):
        return dpnp.matmul(self, other)

 # '__rmod__',

    def __rmul__(self, other):
        return dpnp.multiply(self, other)

 # '__ror__',
 # '__rpow__',
 # '__rrshift__',
 # '__rshift__',
 # '__rsub__',
 # '__rtruediv__',
 # '__rxor__',
 # '__setattr__',

    def __setitem__(self, key, val):
        self._array_obj.__setitem__(key, val)

 # '__setstate__',
 # '__sizeof__',
 # '__str__',
 # '__sub__',
 # '__subclasshook__',
 # '__truediv__',
 # '__xor__',

    def all(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.

        Refer to `numpy.all` for full documentation.

        See Also
        --------
        :obj:`numpy.all` : equivalent function

        """

        return dpnp.all(self, axis, out, keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        """
        Returns True if any of the elements of `a` evaluate to True.

        Refer to `numpy.any` for full documentation.

        See Also
        --------
        :obj:`numpy.any` : equivalent function

        """

        return dpnp.any(self, axis, out, keepdims)

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

        return dpnp.astype(self, dtype, order, casting, subok, copy)

 # 'base',
 # 'byteswap',
 # 'choose',
 # 'clip',
 # 'compress',
 # 'conj',
 # 'conjugate',
 # 'copy',
 # 'ctypes',
 # 'cumprod',
 # 'cumsum',
 # 'data',
 # 'diagonal',
 # 'dot',

    @property
    def dtype(self):
        """
        """

        return self._array_obj.dtype

 # 'dump',
 # 'dumps',
 # 'fill',
 # 'flags',
 # 'flat',
 # 'flatten',
 # 'getfield',
 # 'imag',
 # 'item',
 # 'itemset',
 # 'itemsize',

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
 # 'prod',
 # 'ptp',
 # 'put',
 # 'ravel',
 # 'real',
 # 'repeat',
 # 'reshape',
 # 'resize',
 # 'round',
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
 # 'squeeze',
 # 'std',
 # 'strides',
 # 'sum',
 # 'swapaxes',
 # 'take',
 # 'tobytes',
 # 'tofile',
 # 'tolist',
 # 'tostring',
 # 'trace',
 # 'transpose',
 # 'var',
 # 'view'
 