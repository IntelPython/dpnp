# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
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

"""
Interface of an ndarray representing a multidimensional tensor of numeric
elements stored in a USM allocation on a SYCL device.

"""

# pylint: disable=invalid-name
# pylint: disable=protected-access

import dpctl.tensor as dpt
import dpctl.tensor._type_utils as dtu
from dpctl.tensor._numpy_helper import AxisError

import dpnp
import dpnp.memory as dpm


def _get_unwrapped_index_key(key):
    """
    Get an unwrapped index key.

    Return a key where each nested instance of DPNP array is unwrapped into
    USM ndarray for further processing in DPCTL advanced indexing functions.

    """

    if isinstance(key, tuple):
        if any(isinstance(x, dpnp_array) for x in key):
            # create a new tuple from the input key with unwrapped DPNP arrays
            return tuple(
                x.get_array() if isinstance(x, dpnp_array) else x for x in key
            )
    elif isinstance(key, dpnp_array):
        return key.get_array()
    return key


# pylint: disable=too-many-public-methods
class dpnp_array:
    """
    An array object represents a multidimensional tensor of numeric elements
    stored in a USM allocation on a SYCL device.

    This is a wrapper around :class:`dpctl.tensor.usm_ndarray` that provides
    methods to be compliant with original NumPy.

    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        shape,
        dtype=None,
        buffer=None,
        offset=0,
        strides=None,
        order="C",
        device=None,
        usm_type="device",
        sycl_queue=None,
    ):
        if order is None:
            order = "C"

        if buffer is not None:
            # expecting to have buffer as dpnp.ndarray and usm_ndarray,
            # or as USM memory allocation
            if isinstance(buffer, dpnp_array):
                buffer = buffer.get_array()

            if dtype is None and hasattr(buffer, "dtype"):
                dtype = buffer.dtype
        else:
            buffer = usm_type

        sycl_queue_normalized = dpnp.get_normalized_queue_device(
            device=device, sycl_queue=sycl_queue
        )

        self._array_obj = dpt.usm_ndarray(
            shape,
            dtype=dtype,
            strides=strides,
            buffer=buffer,
            offset=offset,
            order=order,
            buffer_ctor_kwargs={"queue": sycl_queue_normalized},
            array_namespace=dpnp,
        )

    def __abs__(self, /):
        r"""Return :math:`|\text{self}|`."""
        return dpnp.abs(self)

    def __add__(self, other, /):
        r"""Return :math:`\text{self + value}`."""
        return dpnp.add(self, other)

    def __and__(self, other, /):
        r"""Return :math:`\text{self & value}`."""
        return dpnp.bitwise_and(self, other)

    def __array__(self, dtype=None, /, *, copy=None):
        """
        NumPy's array protocol method to disallow implicit conversion.

        Without this definition, ``numpy.asarray(dpnp_arr)`` converts
        :class:`dpnp.ndarray` instance into NumPy array with data type `object`
        and every element being zero-dimensional :class:`dpnp.ndarray`.

        """  # noqa: D403

        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Please use `.asnumpy()` to construct a NumPy array explicitly."
        )

    # '__array_finalize__',
    # '__array_function__',
    # '__array_interface__',

    def __array_namespace__(self, /, *, api_version=None):
        """
        Return array namespace, member functions of which implement data API.

        Parameters
        ----------
        api_version : {None, str}, optional
            Request namespace compliant with given version of array API. If
            ``None``, namespace for the most recent supported version is
            returned.

            Default: ``None``.

        Returns
        -------
        out : any
            An object representing the array API namespace. It should have
            every top-level function defined in the specification as
            an attribute. It may contain other public names as well, but it is
            recommended to only include those names that are part of the
            specification.

        """

        return self._array_obj.__array_namespace__(api_version=api_version)

    # '__array_prepare__',
    # '__array_priority__',
    # '__array_struct__',

    __array_ufunc__ = None

    # '__array_wrap__',

    def __bool__(self, /):
        """``True`` if `self` else ``False``."""
        return self._array_obj.__bool__()

    # '__class__',
    # `__class_getitem__`,

    def __complex__(self, /):
        """Convert a zero-dimensional array to a Python complex object."""
        return self._array_obj.__complex__()

    def __contains__(self, value, /):
        r"""Return :math:`\text{value in self}`."""
        return (self == value).any()

    def __copy__(self):
        """
        Used if :func:`copy.copy` is called on an array. Return a copy of the
        array.

        Equivalent to ``a.copy(order="K")``.

        """
        return self.copy(order="K")

    # '__deepcopy__',
    # '__delattr__',
    # '__delitem__',
    # '__dir__',
    # '__divmod__',

    def __dlpack__(
        self, /, *, stream=None, max_version=None, dl_device=None, copy=None
    ):
        """
        Produce DLPack capsule.

        Parameters
        ----------
        stream : {:class:`dpctl.SyclQueue`, None}, optional
            Execution queue to synchronize with. If ``None``, synchronization
            is not performed.

            Default: ``None``.
        max_version : {tuple of ints, None}, optional
            The maximum DLPack version the consumer (caller of ``__dlpack__``)
            supports. As ``__dlpack__`` may not always return a DLPack capsule
            with version `max_version`, the consumer must verify the version
            even if this argument is passed.

            Default: ``None``.
        dl_device : {tuple, None}, optional:
            The device the returned DLPack capsule will be placed on. The
            device must be a 2-tuple matching the format of
            :meth:`dpnp.ndarray.__dlpack_device__`, an integer enumerator
            representing the device type followed by an integer representing
            the index of the device.

            Default: ``None``.
        copy : {bool, None}, optional:
            Boolean indicating whether or not to copy the input.

            * If `copy` is ``True``, the input will always be copied.
            * If ``False``, a ``BufferError`` will be raised if a copy is
              deemed necessary.
            * If ``None``, a copy will be made only if deemed necessary,
              otherwise, the existing memory buffer will be reused.

            Default: ``None``.

        Raises
        ------
        MemoryError
            when host memory can not be allocated.
        DLPackCreationError
            when array is allocated on a partitioned SYCL device, or with
            a non-default context.
        BufferError
            when a copy is deemed necessary but `copy` is ``False`` or when
            the provided `dl_device` cannot be handled.

        """

        return self._array_obj.__dlpack__(
            stream=stream,
            max_version=max_version,
            dl_device=dl_device,
            copy=copy,
        )

    def __dlpack_device__(self, /):
        """
        Give a tuple (``device_type``, ``device_id``) corresponding to
        ``DLDevice`` entry in ``DLTensor`` in DLPack protocol.

        The tuple describes the non-partitioned device where the array has been
        allocated, or the non-partitioned parent device of the allocation
        device.

        See :class:`dpnp.DLDeviceType` for a list of devices supported by the
        DLPack protocol.

        Raises
        ------
        DLPackCreationError
            when the ``device_id`` could not be determined.

        """

        return self._array_obj.__dlpack_device__()

    # '__doc__',

    def __eq__(self, other, /):
        r"""Return :math:`\text{self == value}`."""
        return dpnp.equal(self, other)

    def __float__(self, /):
        """Convert a zero-dimensional array to a Python float object."""
        return self._array_obj.__float__()

    def __floordiv__(self, other, /):
        r"""Return :math:`\text{self // value}`."""
        return dpnp.floor_divide(self, other)

    # '__format__',

    def __ge__(self, other, /):
        r"""Return :math:`\text{self >= value}`."""
        return dpnp.greater_equal(self, other)

    # '__getattribute__',

    def __getitem__(self, key, /):
        r"""Return :math:`\text{self[key]}`."""
        key = _get_unwrapped_index_key(key)

        item = self._array_obj.__getitem__(key)
        return dpnp_array._create_from_usm_ndarray(item)

    # '__getstate__',

    def __gt__(self, other, /):
        r"""Return :math:`\text{self > value}`."""
        return dpnp.greater(self, other)

    # '__hash__',

    def __iadd__(self, other, /):
        r"""Return :math:`\text{self += value}`."""
        dpnp.add(self, other, out=self)
        return self

    def __iand__(self, other, /):
        r"""Return :math:`\text{self &= value}`."""
        dpnp.bitwise_and(self, other, out=self)
        return self

    def __ifloordiv__(self, other, /):
        r"""Return :math:`\text{self //= value}`."""
        dpnp.floor_divide(self, other, out=self)
        return self

    def __ilshift__(self, other, /):
        r"""Return :math:`\text{self <<= value}`."""
        dpnp.left_shift(self, other, out=self)
        return self

    def __imatmul__(self, other, /):
        r"""Return :math:`\text{self @= value}`."""

        # Unlike `matmul(a, b, out=a)` we ensure that the result isn't broadcast
        # if the result without `out` would have less dimensions than `a`.
        # Since the signature of matmul is '(n?,k),(k,m?)->(n?,m?)' this is the
        # case exactly when the second operand has both core dimensions.
        # We have to enforce this check by passing the correct `axes=`.
        if self.ndim == 1:
            axes = [(-1,), (-2, -1), (-1,)]
        else:
            axes = [(-2, -1), (-2, -1), (-2, -1)]

        try:
            dpnp.matmul(self, other, out=self, dtype=self.dtype, axes=axes)
        except AxisError as e:
            # AxisError should indicate that the axes argument didn't work out
            # which should mean the second operand not being 2 dimensional.
            raise ValueError(
                "inplace matrix multiplication requires the first operand to "
                "have at least one and the second at least two dimensions."
            ) from e
        return self

    def __imod__(self, other, /):
        r"""Return :math:`\text{self %= value}`."""
        dpnp.remainder(self, other, out=self)
        return self

    def __imul__(self, other, /):
        r"""Return :math:`\text{self *= value}`."""
        dpnp.multiply(self, other, out=self)
        return self

    def __index__(self, /):
        """Convert a zero-dimensional array to a Python int object."""
        return self._array_obj.__index__()

    # '__init__',
    # '__init_subclass__',

    def __int__(self, /):
        """Convert a zero-dimensional array to a Python int object."""
        return self._array_obj.__int__()

    def __invert__(self, /):
        r"""Return :math:`\text{~self}`."""
        return dpnp.invert(self)

    def __ior__(self, other, /):
        r"""Return :math:`\text{self |= value}`."""
        dpnp.bitwise_or(self, other, out=self)
        return self

    def __ipow__(self, other, /):
        r"""Return :math:`\text{self **= value}`."""
        dpnp.power(self, other, out=self)
        return self

    def __irshift__(self, other, /):
        r"""Return :math:`\text{self >>= value}`."""
        dpnp.right_shift(self, other, out=self)
        return self

    def __isub__(self, other, /):
        r"""Return :math:`\text{self -= value}`."""
        dpnp.subtract(self, other, out=self)
        return self

    def __iter__(self, /):
        r"""Return :math:`\text{iter(self)}`."""
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        return (self[i] for i in range(self.shape[0]))

    def __itruediv__(self, other, /):
        r"""Return :math:`\text{self /= value}`."""
        dpnp.true_divide(self, other, out=self)
        return self

    def __ixor__(self, other, /):
        r"""Return :math:`\text{self ^= value}`."""
        dpnp.bitwise_xor(self, other, out=self)
        return self

    def __le__(self, other, /):
        r"""Return :math:`\text{self <= value}`."""
        return dpnp.less_equal(self, other)

    def __len__(self):
        r"""Return :math:`\text{len(self)}`."""
        return self._array_obj.__len__()

    def __lshift__(self, other, /):
        r"""Return :math:`\text{self << value}`."""
        return dpnp.left_shift(self, other)

    def __lt__(self, other, /):
        r"""Return :math:`\text{self < value}`."""
        return dpnp.less(self, other)

    def __matmul__(self, other, /):
        r"""Return :math:`\text{self @ value}`."""
        return dpnp.matmul(self, other)

    def __mod__(self, other, /):
        r"""Return :math:`\text{self % value}`."""
        return dpnp.remainder(self, other)

    def __mul__(self, other, /):
        r"""Return :math:`\text{self * value}`."""
        return dpnp.multiply(self, other)

    def __ne__(self, other, /):
        r"""Return :math:`\text{self != value}`."""
        return dpnp.not_equal(self, other)

    def __neg__(self, /):
        r"""Return :math:`\text{-self}`."""
        return dpnp.negative(self)

    # '__new__',

    def __or__(self, other, /):
        r"""Return :math:`\text{self | value}`."""
        return dpnp.bitwise_or(self, other)

    def __pos__(self, /):
        r"""Return :math:`\text{+self}`."""
        return dpnp.positive(self)

    def __pow__(self, other, mod=None, /):
        r"""Return :math:`\text{self ** value}`."""
        if mod is not None:
            return NotImplemented
        return dpnp.power(self, other)

    def __radd__(self, other, /):
        r"""Return :math:`\text{value + self}`."""
        return dpnp.add(other, self)

    def __rand__(self, other, /):
        r"""Return :math:`\text{value & self}`."""
        return dpnp.bitwise_and(other, self)

    # '__rdivmod__',
    # '__reduce__',
    # '__reduce_ex__',

    def __repr__(self):
        r"""Return :math:`\text{repr(self)}`."""
        return dpt.usm_ndarray_repr(self._array_obj, prefix="array")

    def __rfloordiv__(self, other, /):
        r"""Return :math:`\text{value // self}`."""
        return dpnp.floor_divide(self, other)

    def __rlshift__(self, other, /):
        r"""Return :math:`\text{value << self}`."""
        return dpnp.left_shift(other, self)

    def __rmatmul__(self, other, /):
        r"""Return :math:`\text{value @ self}`."""
        return dpnp.matmul(other, self)

    def __rmod__(self, other, /):
        r"""Return :math:`\text{value % self}`."""
        return dpnp.remainder(other, self)

    def __rmul__(self, other, /):
        r"""Return :math:`\text{value * self}`."""
        return dpnp.multiply(other, self)

    def __ror__(self, other, /):
        r"""Return :math:`\text{value | self}`."""
        return dpnp.bitwise_or(other, self)

    def __rpow__(self, other, mod=None, /):
        r"""Return :math:`\text{value ** self}`."""
        if mod is not None:
            return NotImplemented
        return dpnp.power(other, self)

    def __rrshift__(self, other, /):
        r"""Return :math:`\text{value >> self}`."""
        return dpnp.right_shift(other, self)

    def __rshift__(self, other, /):
        r"""Return :math:`\text{self >> value}`."""
        return dpnp.right_shift(self, other)

    def __rsub__(self, other, /):
        r"""Return :math:`\text{value - self}`."""
        return dpnp.subtract(other, self)

    def __rtruediv__(self, other, /):
        r"""Return :math:`\text{value / self}`."""
        return dpnp.true_divide(other, self)

    def __rxor__(self, other, /):
        r"""Return :math:`\text{value ^ self}`."""
        return dpnp.bitwise_xor(other, self)

    # '__setattr__',

    def __setitem__(self, key, value, /):
        r"""Set :math:`\text{self[key]}` to a value."""
        key = _get_unwrapped_index_key(key)

        if isinstance(value, dpnp_array):
            value = value.get_array()

        self._array_obj.__setitem__(key, value)

    # '__setstate__',
    # '__sizeof__',

    __slots__ = ("_array_obj",)

    def __str__(self):
        r"""Return :math:`\text{str(self)}`."""
        return self._array_obj.__str__()

    def __sub__(self, other, /):
        r"""Return :math:`\text{self - value}`."""
        return dpnp.subtract(self, other)

    # '__subclasshook__',

    @property
    def __sycl_usm_array_interface__(self):
        """
        Give ``__sycl_usm_array_interface__`` dictionary describing the array.

        """  # noqa: D200
        return self._array_obj.__sycl_usm_array_interface__

    def __truediv__(self, other, /):
        r"""Return :math:`\text{self / value}`."""
        return dpnp.true_divide(self, other)

    @property
    def __usm_ndarray__(self):
        """
        Property to support ``__usm_ndarray__`` protocol.

        It assumes to return :class:`dpctl.tensor.usm_ndarray` instance
        corresponding to the content of the object.

        This property is intended to speed-up conversion from
        :class:`dpnp.ndarray` to :class:`dpctl.tensor.usm_ndarray` passed into
        :func:`dpctl.tensor.asarray` function. The input object that implements
        ``__usm_ndarray__`` protocol is recognized as owner of USM allocation
        that is managed by a smart pointer, and asynchronous deallocation
        will not involve GIL.

        """

        return self._array_obj

    def __xor__(self, other, /):
        r"""Return :math:`\text{self ^ value}`."""
        return dpnp.bitwise_xor(self, other)

    @staticmethod
    def _create_from_usm_ndarray(usm_ary: dpt.usm_ndarray):
        """
        Return :class:`dpnp.ndarray` instance from USM allocation providing
        by an instance of :class:`dpctl.tensor.usm_ndarray`.

        """

        if not isinstance(usm_ary, dpt.usm_ndarray):
            raise TypeError(
                f"Expected dpctl.tensor.usm_ndarray, got {type(usm_ary)}"
            )
        res = dpnp_array.__new__(dpnp_array)
        res._array_obj = usm_ary
        res._array_obj._set_namespace(dpnp)
        return res

    def all(self, axis=None, *, out=None, keepdims=False, where=True):
        """
        Return ``True`` if all elements evaluate to ``True``.

        Refer to :obj:`dpnp.all` for full documentation.

        See Also
        --------
        :obj:`dpnp.all` : equivalent function

        """

        return dpnp.all(
            self, axis=axis, out=out, keepdims=keepdims, where=where
        )

    def any(self, axis=None, *, out=None, keepdims=False, where=True):
        """
        Return ``True`` if any of the elements of `a` evaluate to ``True``.

        Refer to :obj:`dpnp.any` for full documentation.

        See Also
        --------
        :obj:`dpnp.any` : equivalent function

        """

        return dpnp.any(
            self, axis=axis, out=out, keepdims=keepdims, where=where
        )

    def argmax(self, /, axis=None, out=None, *, keepdims=False):
        """
        Return array of indices of the maximum values along the given axis.

        Refer to :obj:`dpnp.argmax` for full documentation.

        """

        return dpnp.argmax(self, axis=axis, out=out, keepdims=keepdims)

    def argmin(self, /, axis=None, out=None, *, keepdims=False):
        """
        Return array of indices to the minimum values along the given axis.

        Refer to :obj:`dpnp.argmin` for full documentation.

        """

        return dpnp.argmin(self, axis=axis, out=out, keepdims=keepdims)

    # 'argpartition',

    def argsort(
        self, axis=-1, kind=None, order=None, *, descending=False, stable=None
    ):
        """
        Return an ndarray of indices that sort the array along the specified
        axis.

        Refer to :obj:`dpnp.argsort` for full documentation.

        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to sort. If ``None``, the array is flattened
            before sorting. The default is ``-1``, which sorts along the last
            axis.

            Default: ``-1``.
        kind : {None, "stable", "mergesort", "radixsort"}, optional
            Sorting algorithm. The default is ``None``, which uses parallel
            merge-sort or parallel radix-sort algorithms depending on the array
            data type.

            Default: ``None``.
        descending : bool, optional
            Sort order. If ``True``, the array must be sorted in descending
            order (by value). If ``False``, the array must be sorted in
            ascending order (by value).

            Default: ``False``.
        stable : {None, bool}, optional
            Sort stability. If ``True``, the returned array will maintain the
            relative order of `a` values which compare as equal. The same
            behavior applies when set to ``False`` or ``None``.
            Internally, this option selects ``kind="stable"``.

            Default: ``None``.

        See Also
        --------
        :obj:`dpnp.sort` : Return a sorted copy of an array.
        :obj:`dpnp.argsort` : Return the indices that would sort an array.
        :obj:`dpnp.lexsort` : Indirect stable sort on multiple keys.
        :obj:`dpnp.searchsorted` : Find elements in a sorted array.
        :obj:`dpnp.partition` : Partial sort.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([3, 1, 2])
        >>> a.argsort()
        array([1, 2, 0])

        >>> a = np.array([[0, 3], [2, 2]])
        >>> a.argsort(axis=0)
        array([[0, 1],
               [1, 0]])

        """

        return dpnp.argsort(
            self, axis, kind, order, descending=descending, stable=stable
        )

    def asnumpy(self):
        """
        Copy content of the array into :class:`numpy.ndarray` instance of
        the same shape and data type.

        Returns
        -------
        out : numpy.ndarray
            An instance of :class:`numpy.ndarray` populated with the array
            content.

        """

        return dpt.asnumpy(self._array_obj)

    def astype(
        self,
        dtype,
        order="K",
        casting="unsafe",
        subok=True,
        copy=True,
        device=None,
    ):
        """
        Copy the array with data type casting.

        Refer to :obj:`dpnp.astype` for full documentation.

        Parameters
        ----------
        dtype : {None, str, dtype object}
            Target data type.
        order : {None, "C", "F", "A", "K"}, optional
            Row-major (C-style) or column-major (Fortran-style) order.
            When `order` is ``"A"``, it uses ``"F"`` if `a` is column-major and
            uses ``"C"`` otherwise. And when `order` is ``"K"``, it keeps
            strides as closely as possible.

            Default: ``"K"``.
        casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
            Controls what kind of data casting may occur. Defaults to
            ``"unsafe"`` for backwards compatibility.

                - "no" means the data types should not be cast at all.
                - "equiv" means only byte-order changes are allowed.
                - "safe" means only casts which can preserve values are allowed.
                - "same_kind" means only safe casts or casts within a kind,
                  like float64 to float32, are allowed.
                - "unsafe" means any data conversions may be done.

            Default: ``"unsafe"``.
        copy : bool, optional
            Specifies whether to copy an array when the specified dtype matches
            the data type of that array. If ``True``, a newly allocated array
            must always be returned. If ``False`` and the specified dtype
            matches the data type of that array, the self array must be
            returned; otherwise, a newly allocated array must be returned.

            Default: ``True``.
        device : {None, string, SyclDevice, SyclQueue, Device}, optional
            An array API concept of device where the output array is created.
            `device` can be ``None``, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a :class:`dpctl.tensor.Device` object
            returned by :attr:`dpnp.ndarray.device`.
            If the value is ``None``, returned array is created on the same
            device as that array.

            Default: ``None``.

        Returns
        -------
        out : dpnp.ndarray
            An array having the specified data type.

        Limitations
        -----------
        Parameter `subok` is supported with default value.
        Otherwise ``NotImplementedError`` exception will be raised.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.array([1, 2, 2.5]); x
        array([1. , 2. , 2.5])

        >>> x.astype(int)
        array([1, 2, 2])

        """

        if subok is not True:
            raise NotImplementedError(
                f"subok={subok} is currently not supported"
            )

        return dpnp.astype(
            self, dtype, order=order, casting=casting, copy=copy, device=device
        )

    # 'base',
    # 'byteswap',

    def choose(self, /, choices, out=None, mode="wrap"):
        """
        Use an array as index array to construct a new array from a set of
        choices.

        Refer to :obj:`dpnp.choose` for full documentation.

        """

        return dpnp.choose(self, choices, out, mode)

    def clip(self, /, min=None, max=None, out=None, **kwargs):
        """
        Clip (limit) the values in an array.

        Refer to :obj:`dpnp.clip` for full documentation.

        """

        return dpnp.clip(self, min, max, out=out, **kwargs)

    def compress(self, /, condition, axis=None, *, out=None):
        """
        Select slices of an array along a given axis.

        Refer to :obj:`dpnp.compress` for full documentation.
        """

        return dpnp.compress(condition, self, axis=axis, out=out)

    def conj(self):
        """
        Complex-conjugate all elements.

        Refer to :obj:`dpnp.conjugate` for full documentation.

        """

        return self.conjugate()

    def conjugate(self):
        """
        Return the complex conjugate, element-wise.

        Refer to :obj:`dpnp.conjugate` for full documentation.

        """

        if not dpnp.issubdtype(self.dtype, dpnp.complexfloating):
            return self
        return dpnp.conjugate(self)

    def copy(
        self, /, order="C", *, device=None, usm_type=None, sycl_queue=None
    ):
        """
        Return a copy of the array.

        Refer to :obj:`dpnp.copy` for full documentation.

        Parameters
        ----------
        order : {None, "C", "F", "A", "K"}, optional
            Memory layout of the newly output array.

            Default: ``"C"``.
        device : {None, string, SyclDevice, SyclQueue, Device}, optional
            An array API concept of device where the output array is created.
            `device` can be ``None``, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a :class:`dpctl.tensor.Device` object
            returned by :attr:`dpnp.ndarray.device`.

            Default: ``None``.
        usm_type : {None, "device", "shared", "host"}, optional
            The type of SYCL USM allocation for the output array.

            Default: ``None``.
        sycl_queue : {None, SyclQueue}, optional
            A SYCL queue to use for output array allocation and copying. The
            `sycl_queue` can be passed as ``None`` (the default), which means
            to get the SYCL queue from `device` keyword if present or to use
            a default queue.

            Default: ``None``.

        Returns
        -------
        out : dpnp.ndarray
            A copy of the array.

        See also
        --------
        :obj:`dpnp.copy` : Similar function with different default behavior
        :obj:`dpnp.copyto` : Copies values from one array to another.

        Notes
        -----
        This function is the preferred method for creating an array copy.
        The function :func:`dpnp.copy` is similar, but it defaults to using
        order ``"K"``.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.array([[1, 2, 3], [4, 5, 6]], order='F')
        >>> y = x.copy()
        >>> x.fill(0)

        >>> x
        array([[0, 0, 0],
               [0, 0, 0]])

        >>> y
        array([[1, 2, 3],
               [4, 5, 6]])

        >>> y.flags['C_CONTIGUOUS']
        True

        """

        return dpnp.copy(
            self,
            order=order,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )

    # 'ctypes',

    def cumprod(self, /, axis=None, dtype=None, *, out=None):
        """
        Return the cumulative product of the elements along the given axis.

        Refer to :obj:`dpnp.cumprod` for full documentation.

        """

        return dpnp.cumprod(self, axis=axis, dtype=dtype, out=out)

    def cumsum(self, /, axis=None, dtype=None, *, out=None):
        """
        Return the cumulative sum of the elements along the given axis.

        Refer to :obj:`dpnp.cumsum` for full documentation.

        """

        return dpnp.cumsum(self, axis=axis, dtype=dtype, out=out)

    @property
    def data(self):
        """
        Python object pointing to the start of USM memory allocation with the
        array's data.

        """

        return dpm.create_data(self._array_obj)

    @property
    def device(self):
        """
        Return :class:`dpctl.tensor.Device` object representing residence of
        the array data.

        The ``Device`` object represents Array API notion of the device, and
        contains :class:`dpctl.SyclQueue` associated with this array. Hence,
        ``.device`` property provides information distinct from ``.sycl_device``
        property.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.ones(10)
        >>> x.device
        Device(level_zero:gpu:0)

        """

        return self._array_obj.device

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """
        Return specified diagonals.

        Refer to :obj:`dpnp.diagonal` for full documentation.

        See Also
        --------
        :obj:`dpnp.diagonal` : Equivalent function.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.arange(4).reshape(2, 2)
        >>> a.diagonal()
        array([0, 3])

        """

        return dpnp.diagonal(self, offset=offset, axis1=axis1, axis2=axis2)

    def dot(self, b, out=None):
        """
        Dot product of two arrays.

        Refer to :obj:`dpnp.dot` for full documentation.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.eye(2)
        >>> b = np.ones((2, 2)) * 2
        >>> a.dot(b)
        array([[2., 2.],
               [2., 2.]])

        This array method can be conveniently chained:

        >>> a.dot(b).dot(b)
        array([[8., 8.],
               [8., 8.]])
        """

        return dpnp.dot(self, b, out)

    @property
    def dtype(self):
        """
        Return NumPy's dtype corresponding to the type of the array elements.

        """  # noqa: D200

        return self._array_obj.dtype

    # 'dump',
    # 'dumps',

    def fill(self, value):
        """
        Fill the array with a scalar value.

        For full documentation refer to :obj:`numpy.ndarray.fill`.

        Parameters
        ----------
        value : {dpnp.ndarray, usm_ndarray, scalar}
            All elements of `a` will be assigned this value.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([1, 2])
        >>> a.fill(0)
        >>> a
        array([0, 0])
        >>> a = np.empty(2)
        >>> a.fill(1)
        >>> a
        array([1.,  1.])

        """

        # lazy import avoids circular imports
        # pylint: disable=import-outside-toplevel
        from .dpnp_algo.dpnp_fill import dpnp_fill

        dpnp_fill(self, value)

    @property
    def flags(self):
        """Return information about the memory layout of the array."""

        return self._array_obj.flags

    @property
    def flat(self):
        """
        Return a flat iterator, or set a flattened version of self to value.

        """  # noqa: D200

        return dpnp.flatiter(self)

    def flatten(self, /, order="C"):
        """
        Return a copy of the array collapsed into one dimension.

        For full documentation refer to :obj:`numpy.ndarray.flatten`.

        Parameters
        ----------
        order : {"C", "F"}, optional
            Read the elements using this index order, and place the elements
            into the reshaped array using this index order.

                - ``"C"`` means to read / write the elements using C-like index
                  order, with the last axis index changing fastest, back to the
                  first axis index changing slowest.
                - ``"F"`` means to read / write the elements using Fortran-like
                  index order, with the first index changing fastest, and the
                  last index changing slowest.

            Default: ``"C"``.

        Returns
        -------
        out : dpnp.ndarray
            A copy of the input array, flattened to one dimension.

        See Also
        --------
        :obj:`dpnp.ravel` : Return a flattened array.
        :obj:`dpnp.flat` : A 1-D flat iterator over the array.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([[1, 2], [3, 4]])
        >>> a.flatten()
        array([1, 2, 3, 4])
        >>> a.flatten("F")
        array([1, 3, 2, 4])

        """

        return self.reshape(-1, order=order, copy=True)

    def get_array(self):
        """Get :class:`dpctl.tensor.usm_ndarray` object."""
        return self._array_obj

    # 'getfield',

    @property
    def imag(self, /):
        """
        The imaginary part of the array.

        For full documentation refer to :obj:`numpy.ndarray.imag`.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.sqrt(np.array([1+0j, 0+1j]))
        >>> x.imag
        array([0.        , 0.70710677])

        """
        return dpnp_array._create_from_usm_ndarray(
            dpnp.get_usm_ndarray(self).imag
        )

    @imag.setter
    def imag(self, value, /):
        """
        Set the imaginary part of the array.

        For full documentation refer to :obj:`numpy.ndarray.imag`.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([1+2j, 3+4j, 5+6j])
        >>> a.imag = 9
        >>> a
        array([1.+9.j, 3.+9.j, 5.+9.j])

        """

        if dpnp.issubdtype(self.dtype, dpnp.complexfloating):
            dpnp.copyto(self._array_obj.imag, value)
        else:
            raise TypeError("array does not have imaginary part to set")

    def item(self, /, *args):
        """
        Copy an element of an array to a standard Python scalar and return it.

        For full documentation refer to :obj:`numpy.ndarray.item`.

        Parameters
        ----------
        *args : {none, int, tuple of ints}
            - none: in this case, the method only works for arrays with
              one element (``a.size == 1``), which element is copied into a
              standard Python scalar object and returned.
            - int: this argument is interpreted as a flat index into the array,
              specifying which element to copy and return.
            - tuple of ints: functions as does a single int argument, except
              that the argument is interpreted as an nd-index into the array.

        Returns
        -------
        out : Standard Python scalar object
            A copy of the specified element of the array as a suitable Python
            scalar.

        Examples
        --------
        >>> import dpnp as np
        >>> np.random.seed(123)
        >>> x = np.random.randint(9, size=(3, 3))
        >>> x
        array([[0, 0, 7],
               [6, 6, 6],
               [0, 7, 1]])
        >>> x.item(3)
        6
        >>> x.item(7)
        7
        >>> x.item((0, 1))
        0
        >>> x.item((2, 2))
        1

        >>> x = np.array(5)
        >>> x.item()
        5

        """

        # TODO: implement a more efficient way to avoid copying to host
        # for large arrays using `asnumpy()`
        return self.asnumpy().item(*args)

    @property
    def itemsize(self):
        """Size of one array element in bytes."""

        return self._array_obj.itemsize

    def max(
        self,
        /,
        axis=None,
        *,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        """
        Return the maximum along an axis.

        Refer to :obj:`dpnp.max` for full documentation.

        """

        return dpnp.max(
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def mean(
        self, /, axis=None, dtype=None, *, out=None, keepdims=False, where=True
    ):
        """
        Return the average of the array elements.

        Refer to :obj:`dpnp.mean` for full documentation.

        """

        return dpnp.mean(self, axis, dtype, out, keepdims, where=where)

    def min(
        self,
        /,
        axis=None,
        *,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        """
        Return the minimum along a given axis.

        Refer to :obj:`dpnp.min` for full documentation.

        """

        return dpnp.min(
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @property
    def mT(self):
        """
        View of the matrix transposed array.

        The matrix transpose is the transpose of the last two dimensions, even
        if the array is of higher dimension.

        Raises
        ------
        ValueError
            If the array is of dimension less than ``2``.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([[1, 2], [3, 4]])
        >>> a
        array([[1, 2],
               [3, 4]])
        >>> a.mT
        array([[1, 3],
               [2, 4]])

        >>> a = np.arange(8).reshape((2, 2, 2))
        >>> a
        array([[[0, 1],
                [2, 3]],
               [[4, 5],
                [6, 7]]])
        >>> a.mT
        array([[[0, 2],
                [1, 3]],
               [[4, 6],
                [5, 7]]])

        """

        if self.ndim < 2:
            raise ValueError("matrix transpose with ndim < 2 is undefined")

        return dpnp_array._create_from_usm_ndarray(self._array_obj.mT)

    @property
    def nbytes(self):
        """Total bytes consumed by the elements of the array."""

        return self._array_obj.nbytes

    @property
    def ndim(self):
        """
        Return the number of dimensions of an array.

        For full documentation refer to :obj:`numpy.ndarray.ndim`.

        Returns
        -------
        number_of_dimensions : int
            The number of dimensions in `a`.

        See Also
        --------
        :obj:`dpnp.ndim` : Equivalent method for any array-like input.
        :obj:`dpnp.shape` : Return the shape of an array.
        :obj:`dpnp.ndarray.shape` : Return the shape of an array.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.array([1, 2, 3])
        >>> x.ndim
        1
        >>> y = np.zeros((2, 3, 4))
        >>> y.ndim
        3

        """

        return self._array_obj.ndim

    def nonzero(self):
        """
        Return the indices of the elements that are non-zero.

        Refer to :obj:`dpnp.nonzero` for full documentation.

        """

        return dpnp.nonzero(self)

    def partition(self, /, kth, axis=-1, kind="introselect", order=None):
        """
        Return a partitioned copy of an array.

        Rearranges the elements in the array in such a way that the value of
        the element in `kth` position is in the position it would be in
        a sorted array.

        All elements smaller than the `kth` element are moved before this
        element and all equal or greater are moved behind it. The ordering
        of the elements in the two partitions is undefined.

        Refer to `dpnp.partition` for full documentation.

        See Also
        --------
        :obj:`dpnp.partition` : Return a partitioned copy of an array.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([3, 4, 2, 1])
        >>> a.partition(3)
        >>> a
        array([1, 2, 3, 4])

        """

        self._array_obj = dpnp.partition(
            self, kth, axis=axis, kind=kind, order=order
        ).get_array()

    def prod(
        self,
        /,
        axis=None,
        dtype=None,
        *,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        """
        Return the prod along a given axis.

        Refer to :obj:`dpnp.prod` for full documentation.

        """

        return dpnp.prod(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def put(self, /, indices, vals, axis=None, mode="wrap"):
        """
        Put values of an array into another array along a given axis.

        Refer to :obj:`dpnp.put` for full documentation.

        """

        return dpnp.put(self, indices, vals, axis=axis, mode=mode)

    def ravel(self, /, order="C"):
        """
        Return a contiguous flattened array.

        Refer to :obj:`dpnp.ravel` for full documentation.

        """

        return dpnp.ravel(self, order=order)

    @property
    def real(self, /):
        """
        The real part of the array.

        For full documentation refer to :obj:`numpy.ndarray.real`.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.sqrt(np.array([1+0j, 0+1j]))
        >>> x.real
        array([1.        , 0.70710677])

        """

        if dpnp.issubdtype(self.dtype, dpnp.complexfloating):
            return dpnp_array._create_from_usm_ndarray(
                dpnp.get_usm_ndarray(self).real
            )
        return self

    @real.setter
    def real(self, value, /):
        """
        Set the real part of the array.

        For full documentation refer to :obj:`numpy.ndarray.real`.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([1+2j, 3+4j, 5+6j])
        >>> a.real = 9
        >>> a
        array([9.+2.j, 9.+4.j, 9.+6.j])

        """

        dpnp.copyto(self._array_obj.real, value)

    def repeat(self, repeats, axis=None):
        """
        Repeat elements of an array.

        Refer to :obj:`dpnp.repeat` for full documentation.

        """

        return dpnp.repeat(self, repeats, axis=axis)

    def reshape(self, /, *shape, order="C", copy=None):
        """
        Return an array containing the same data with a new shape.

        Refer to :obj:`dpnp.reshape` for full documentation.

        Returns
        -------
        y : dpnp.ndarray
            This will be a new view object if possible;
            otherwise, it will be a copy.

        See Also
        --------
        :obj:`dpnp.reshape` : Equivalent function.

        Notes
        -----
        Unlike the free function `dpnp.reshape`, this method on `ndarray` allows
        the elements of the shape parameter to be passed in as separate
        arguments.
        For example, ``a.reshape(10, 11)`` is equivalent to
        ``a.reshape((10, 11))``.

        """

        if len(shape) == 1:
            shape = shape[0]
        return dpnp.reshape(self, shape, order=order, copy=copy)

    # 'resize',

    def round(self, /, decimals=0, *, out=None):
        """
        Return array with each element rounded to the given number of decimals.

        Refer to :obj:`dpnp.round` for full documentation.

        """

        return dpnp.around(self, decimals, out)

    def searchsorted(self, v, side="left", sorter=None):
        """
        Find indices where elements of `v` should be inserted in `a`
        to maintain order.

        Refer to :obj:`dpnp.searchsorted` for full documentation

        """

        return dpnp.searchsorted(self, v, side=side, sorter=sorter)

    # 'setfield',
    # 'setflags',

    @property
    def shape(self):
        """
        Tuple of array dimensions.

        The shape property is usually used to get the current shape of an array,
        but may also be used to reshape the array in-place by assigning a tuple
        of array dimensions to it. Unlike :obj:`dpnp.reshape`, only non-negative
        values are supported to be set as new shape. Reshaping an array in-place
        will fail if a copy is required.

        For full documentation refer to :obj:`numpy.ndarray.shape`.

        Note
        ----
        Using :obj:`dpnp.ndarray.reshape` or :obj:`dpnp.reshape` is the
        preferred approach to set new shape of an array.

        See Also
        --------
        :obj:`dpnp.shape` : Equivalent getter function.
        :obj:`dpnp.reshape` : Function similar to setting `shape`.
        :obj:`dpnp.ndarray.reshape` : Method similar to setting `shape`.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.array([1, 2, 3, 4])
        >>> x.shape
        (4,)
        >>> y = np.zeros((2, 3, 4))
        >>> y.shape
        (2, 3, 4)

        >>> y.shape = (3, 8)
        >>> y
        array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        >>> y.shape = (3, 6)
        ...
        TypeError: Can not reshape array of size 24 into (3, 6)

        """

        return self._array_obj.shape

    @shape.setter
    def shape(self, newshape):
        """
        Set new lengths of axes.

        Modifies array instance in-place by changing its metadata about the
        shape and the strides of the array, or raises `AttributeError`
        exception if in-place change is not possible.

        Whether the array can be reshape in-place depends on its strides. Use
        :obj:`dpnp.reshape` function which always succeeds to reshape the array
        by performing a copy if necessary.

        For full documentation refer to :obj:`numpy.ndarray.shape`.

        Parameters
        ----------
        newshape : {tuple, int}
            New shape. Only non-negative values are supported. The new shape
            may not lead to the change in the number of elements in the array.

        """

        self._array_obj.shape = newshape

    @property
    def size(self):
        """
        Number of elements in the array.

        Returns
        -------
        element_count : int
            Number of elements in the array.

        See Also
        --------
        :obj:`dpnp.size` : Return the number of elements along a given axis.
        :obj:`dpnp.shape` : Return the shape of an array.
        :obj:`dpnp.ndarray.shape` : Return the shape of an array.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.zeros((3, 5, 2), dtype=np.complex64)
        >>> x.size
        30

        """

        return self._array_obj.size

    def sort(
        self, axis=-1, kind=None, order=None, *, descending=False, stable=None
    ):
        """
        Sort an array in-place.

        Refer to :obj:`dpnp.sort` for full documentation.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. The default is ``-1``, which sorts along
            the last axis.

            Default: ``-1``.
        kind : {None, "stable", "mergesort", "radixsort"}, optional
            Sorting algorithm. The default is ``None``, which uses parallel
            merge-sort or parallel radix-sort algorithms depending on the array
            data type.

            Default: ``None``.
        descending : bool, optional
            Sort order. If ``True``, the array must be sorted in descending
            order (by value). If ``False``, the array must be sorted in
            ascending order (by value).

            Default: ``False``.
        stable : {None, bool}, optional
            Sort stability. If ``True``, the returned array will maintain the
            relative order of `a` values which compare as equal. The same
            behavior applies when set to ``False`` or ``None``.
            Internally, this option selects ``kind="stable"``.

            Default: ``None``.

        See Also
        --------
        :obj:`dpnp.sort` : Return a sorted copy of an array.
        :obj:`dpnp.argsort` : Return the indices that would sort an array.
        :obj:`dpnp.lexsort` : Indirect stable sort on multiple keys.
        :obj:`dpnp.searchsorted` : Find elements in a sorted array.
        :obj:`dpnp.partition` : Partial sort.

        Note
        ----
        `axis` in :obj:`dpnp.sort` could be integer or ``None``. If ``None``,
        the array is flattened before sorting. However, `axis` in
        :obj:`dpnp.ndarray.sort` can only be integer since it sorts an array
        in-place.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([[1, 4], [3, 1]])
        >>> a.sort(axis=1)
        >>> a
        array([[1, 4],
              [1, 3]])
        >>> a.sort(axis=0)
        >>> a
        array([[1, 1],
              [3, 4]])

        """

        if axis is None:
            raise TypeError(
                "'NoneType' object cannot be interpreted as an integer"
            )
        self[...] = dpnp.sort(
            self,
            axis=axis,
            kind=kind,
            order=order,
            descending=descending,
            stable=stable,
        )

    def squeeze(self, axis=None):
        """
        Remove single-dimensional entries from the shape of an array.

        Refer to :obj:`dpnp.squeeze` for full documentation

        """

        return dpnp.squeeze(self, axis)

    def std(
        self,
        axis=None,
        dtype=None,
        *,
        out=None,
        ddof=0,
        keepdims=False,
        where=True,
        mean=None,
        correction=None,
    ):
        """
        Return the standard deviation of the array elements, along given axis.

        Refer to :obj:`dpnp.std` for full documentation.

        """

        return dpnp.std(
            self,
            axis,
            dtype,
            out,
            ddof,
            keepdims,
            where=where,
            mean=mean,
            correction=correction,
        )

    @property
    def strides(self):
        """
        Return memory displacement in array elements, upon unit
        change of respective index.

        For example, for strides ``(s1, s2, s3)`` and multi-index
        ``(i1, i2, i3)`` position of the respective element relative
        to zero multi-index element is ``s1*s1 + s2*i2 + s3*i3``.

        """

        return self._array_obj.strides

    def sum(
        self,
        /,
        axis=None,
        dtype=None,
        *,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        """
        Return the sum along a given axis.

        Refer to :obj:`dpnp.sum` for full documentation.

        """

        return dpnp.sum(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def swapaxes(self, axis1, axis2):
        """
        Interchange two axes of an array.

        Refer to :obj:`dpnp.swapaxes` for full documentation.

        """

        return dpnp.swapaxes(self, axis1=axis1, axis2=axis2)

    @property
    def sycl_context(self):
        """
        Return :class:`dpctl.SyclContext` object to which USM data is bound.

        """  # noqa: D200
        return self._array_obj.sycl_context

    @property
    def sycl_device(self):
        """
        Return :class:`dpctl.SyclDevice` object on which USM data was
        allocated.

        """
        return self._array_obj.sycl_device

    @property
    def sycl_queue(self):
        """
        Return :class:`dpctl.SyclQueue` object associated with USM data.

        """  # noqa: D200
        return self._array_obj.sycl_queue

    @property
    def T(self):
        """
        View of the transposed array.

        Same as ``self.transpose()``.

        See Also
        --------
        :obj:`dpnp.transpose` : Equivalent function.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([[1, 2], [3, 4]])
        >>> a
        array([[1, 2],
            [3, 4]])
        >>> a.T
        array([[1, 3],
            [2, 4]])

        >>> a = np.array([1, 2, 3, 4])
        >>> a
        array([1, 2, 3, 4])
        >>> a.T
        array([1, 2, 3, 4])

        """

        return self.transpose()

    def take(self, indices, axis=None, *, out=None, mode="wrap"):
        """
        Take elements from an array along an axis.

        Refer to :obj:`dpnp.take` for full documentation.

        """

        return dpnp.take(self, indices, axis=axis, out=out, mode=mode)

    def to_device(self, device, /, *, stream=None):
        """
        Transfer this array to specified target device.

        Parameters
        ----------
        device : {None, string, SyclDevice, SyclQueue, Device}, optional
            An array API concept of device where the output array is created.
            `device` can be ``None``, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a :class:`dpctl.tensor.Device` object
            returned by :attr:`dpnp.ndarray.device`.
        stream : {SyclQueue, None}, optional
            Execution queue to synchronize with. If ``None``, synchronization
            is not performed.

            Default: ``None``.

        Returns
        -------
        out : dpnp.ndarray
            A view if data copy is not required, and a copy otherwise.
            If copying is required, it is done by copying from the original
            allocation device to the host, followed by copying from host
            to the target device.

        Examples
        --------
        >>> import dpnp as np, dpctl
        >>> x = np.full(100, 2, dtype=np.int64)
        >>> q_prof = dpctl.SyclQueue(x.sycl_device, property="enable_profiling")
        >>> # return a view with profile-enabled queue
        >>> y = x.to_device(q_prof)
        >>> timer = dpctl.SyclTimer()
        >>> with timer(q_prof):
        ...     z = y * y
        >>> print(timer.dt)

        """

        usm_res = self._array_obj.to_device(device, stream=stream)
        return dpnp_array._create_from_usm_ndarray(usm_res)

    # 'tobytes',
    # 'tofile',
    # 'tolist',

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, *, out=None):
        """
        Return the sum along diagonals of the array.

        Refer to :obj:`dpnp.trace` for full documentation.

        """

        return dpnp.trace(
            self, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
        )

    def transpose(self, *axes):
        """
        Return a view of the array with axes transposed.

        For full documentation refer to :obj:`numpy.ndarray.transpose`.

        Parameters
        ----------
        axes : None, tuple or list of ints, n ints, optional
            * ``None`` or no argument: reverses the order of the axes.
            * ``tuple or list of ints``: `i` in the `j`-th place in the
              tuple/list means that the array’s `i`-th axis becomes the
              transposed array’s `j`-th axis.
            * ``n ints``: same as an n-tuple/n-list of the same integers (this
              form is intended simply as a “convenience” alternative to the
              tuple form).

        Returns
        -------
        out : dpnp.ndarray
            View of the array with its axes suitably permuted.

        See Also
        --------
        :obj:`dpnp.transpose` : Equivalent function.
        :obj:`dpnp.ndarray.ndarray.T` : Array property returning the array
            transposed.
        :obj:`dpnp.ndarray.reshape` : Give a new shape to an array without
            changing its data.

        Examples
        --------
        >>> import dpnp as np
        >>> a = np.array([[1, 2], [3, 4]])
        >>> a
        array([[1, 2],
               [3, 4]])
        >>> a.transpose()
        array([[1, 3],
               [2, 4]])
        >>> a.transpose((1, 0))
        array([[1, 3],
               [2, 4]])

        >>> a = np.array([1, 2, 3, 4])
        >>> a
        array([1, 2, 3, 4])
        >>> a.transpose()
        array([1, 2, 3, 4])

        """

        ndim = self.ndim
        if ndim < 2:
            return self

        axes_len = len(axes)
        if axes_len == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]

        if ndim == 2 and axes_len == 0:
            usm_res = self._array_obj.T
        else:
            if len(axes) == 0 or axes[0] is None:
                # self.transpose().shape == self.shape[::-1]
                # self.transpose(None).shape == self.shape[::-1]
                axes = tuple((ndim - x - 1) for x in range(ndim))

            usm_res = dpt.permute_dims(self._array_obj, axes)
        return dpnp_array._create_from_usm_ndarray(usm_res)

    def var(
        self,
        axis=None,
        dtype=None,
        *,
        out=None,
        ddof=0,
        keepdims=False,
        where=True,
        mean=None,
        correction=None,
    ):
        """
        Return the variance of the array elements, along given axis.

        Refer to :obj:`dpnp.var` for full documentation.

        """

        return dpnp.var(
            self,
            axis,
            dtype,
            out,
            ddof,
            keepdims,
            where=where,
            mean=mean,
            correction=correction,
        )

    def view(self, /, dtype=None, *, type=None):
        """
        New view of array with the same data.

        For full documentation refer to :obj:`numpy.ndarray.view`.

        Parameters
        ----------
        dtype : {None, str, dtype object}, optional
            The desired data type of the returned view, e.g. :obj:`dpnp.float32`
            or :obj:`dpnp.int16`. By default, it results in the view having the
            same data type.

            Default: ``None``.

        Notes
        -----
        Passing ``None`` for `dtype` is the same as omitting the parameter,
        opposite to NumPy where they have different meaning.

        ``view(some_dtype)`` or ``view(dtype=some_dtype)`` constructs a view of
        the array's memory with a different data type. This can cause a
        reinterpretation of the bytes of memory.

        Only the last axis has to be contiguous.

        Limitations
        -----------
        Parameter `type` is supported only with default value ``None``.
        Otherwise, the function raises ``NotImplementedError`` exception.

        Examples
        --------
        >>> import dpnp as np
        >>> x = np.ones((4,), dtype=np.float32)
        >>> xv = x.view(dtype=np.int32)
        >>> xv[:] = 0
        >>> xv
        array([0, 0, 0, 0], dtype=int32)

        However, views that change dtype are totally fine for arrays with a
        contiguous last axis, even if the rest of the axes are not C-contiguous:

        >>> x = np.arange(2 * 3 * 4, dtype=np.int8).reshape(2, 3, 4)
        >>> x.transpose(1, 0, 2).view(np.int16)
        array([[[ 256,  770],
                [3340, 3854]],
        <BLANKLINE>
            [[1284, 1798],
                [4368, 4882]],
        <BLANKLINE>
            [[2312, 2826],
                [5396, 5910]]], dtype=int16)

        """

        if type is not None:
            raise NotImplementedError(
                "Keyword argument `type` is supported only with "
                f"default value ``None``, but got {type}."
            )

        old_sh = self.shape
        old_strides = self.strides

        if dtype is None:
            return dpnp_array(old_sh, buffer=self, strides=old_strides)

        new_dt = dpnp.dtype(dtype)
        new_dt = dtu._to_device_supported_dtype(new_dt, self.sycl_device)

        new_itemsz = new_dt.itemsize
        old_itemsz = self.dtype.itemsize
        if new_itemsz == old_itemsz:
            return dpnp_array(
                old_sh, dtype=new_dt, buffer=self, strides=old_strides
            )

        ndim = self.ndim
        if ndim == 0:
            raise ValueError(
                "Changing the dtype of a 0d array is only supported "
                "if the itemsize is unchanged"
            )

        # resize on last axis only
        axis = ndim - 1
        if old_sh[axis] != 1 and self.size != 0 and old_strides[axis] != 1:
            raise ValueError(
                "To change to a dtype of a different size, "
                "the last axis must be contiguous"
            )

        # normalize strides whenever itemsize changes
        if old_itemsz > new_itemsz:
            new_strides = list(
                el * (old_itemsz // new_itemsz) for el in old_strides
            )
        else:
            new_strides = list(
                el // (new_itemsz // old_itemsz) for el in old_strides
            )
        new_strides[axis] = 1
        new_strides = tuple(new_strides)

        new_dim = old_sh[axis] * old_itemsz
        if new_dim % new_itemsz != 0:
            raise ValueError(
                "When changing to a larger dtype, its size must be a divisor "
                "of the total size in bytes of the last axis of the array"
            )

        # normalize shape whenever itemsize changes
        new_sh = list(old_sh)
        new_sh[axis] = new_dim // new_itemsz
        new_sh = tuple(new_sh)

        return dpnp_array(
            new_sh,
            dtype=new_dt,
            buffer=self,
            strides=new_strides,
        )

    @property
    def usm_type(self):
        """
        USM type of underlying memory. Possible values are:

        * ``"device"``
            USM-device allocation in device memory, only accessible to kernels
            executed on the device
        * ``"shared"``
            USM-shared allocation in device memory, accessible both from the
            device and from the host
        * ``"host"``
            USM-host allocation in host memory, accessible both from the device
            and from the host

        """

        return self._array_obj.usm_type
