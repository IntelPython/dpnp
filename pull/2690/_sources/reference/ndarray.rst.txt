Multi-Dimensional Array (ndarray)
=================================

:class:`dpnp.ndarray` is the DPNP counterpart of NumPy :class:`numpy.ndarray`.

For the basic concept of ``ndarray``\s, please refer to the `NumPy documentation <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray
   dpnp.dpnp_array.dpnp_array


Constructing arrays
-------------------

New arrays can be constructed using the routines detailed in
:ref:`Array Creation Routines <routines.array-creation>`, and also by using the low-level
:class:`dpnp.ndarray` constructor:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray


Indexing arrays
---------------

Arrays can be indexed using an extended Python slicing syntax,
``array[selection]``.

.. seealso:: :ref:`Indexing routines <routines.indexing>`.


Array attributes
----------------

Array attributes reflect information that is intrinsic to the array
itself. Generally, accessing an array through its attributes allows
you to get and sometimes set intrinsic properties of the array without
creating a new array. The exposed attributes are the core parts of an
array and only some of them can be reset meaningfully without creating
a new array. Information on each attribute is given below.


Memory layout
-------------

The following attributes contain information about the memory layout
of the array:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.flags
   dpnp.ndarray.shape
   dpnp.ndarray.strides
   dpnp.ndarray.ndim
   dpnp.ndarray.data
   dpnp.ndarray.size
   dpnp.ndarray.itemsize
   dpnp.ndarray.nbytes
   dpnp.ndarray.device
   dpnp.ndarray.sycl_context
   dpnp.ndarray.sycl_device
   dpnp.ndarray.sycl_queue
   dpnp.ndarray.usm_type


Data type
---------

.. seealso:: :ref:`Available array data types <Data types>`

The data type object associated with the array can be found in the
:attr:`dtype <dpnp.ndarray.dtype>` attribute:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.dtype


Other attributes
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.T
   dpnp.ndarray.mT
   dpnp.ndarray.real
   dpnp.ndarray.imag
   dpnp.ndarray.flat


Special attributes
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__sycl_usm_array_interface__
   dpnp.ndarray.__usm_ndarray__


Array methods
-------------

An :class:`dpnp.ndarray` object has many methods which operate on or with
the array in some fashion, typically returning an array result. These
methods are briefly explained below. (Each method's docstring has a
more complete description.)

For the following methods there are also corresponding functions in
:mod:`dpnp`: :func:`all <dpnp.all>`, :func:`any <dpnp.any>`,
:func:`argmax <dpnp.argmax>`, :func:`argmin <dpnp.argmin>`,
:func:`argpartition <dpnp.argpartition>`, :func:`argsort <dpnp.argsort>`,
:func:`choose <dpnp.choose>`, :func:`clip <dpnp.clip>`,
:func:`compress <dpnp.compress>`, :func:`copy <dpnp.copy>`,
:func:`cumprod <dpnp.cumprod>`, :func:`cumsum <dpnp.cumsum>`,
:func:`diagonal <dpnp.diagonal>`, :func:`imag <dpnp.imag>`,
:func:`max <dpnp.max>`, :func:`mean <dpnp.mean>`, :func:`min <dpnp.min>`,
:func:`nonzero <dpnp.nonzero>`, :func:`partition <dpnp.partition>`,
:func:`prod <dpnp.prod>`, :func:`put <dpnp.put>`,
:func:`ravel <dpnp.ravel>`, :func:`real <dpnp.real>`, :func:`repeat <dpnp.repeat>`,
:func:`reshape <dpnp.reshape>`, :func:`round <dpnp.around>`,
:func:`searchsorted <dpnp.searchsorted>`, :func:`sort <dpnp.sort>`,
:func:`squeeze <dpnp.squeeze>`, :func:`std <dpnp.std>`, :func:`sum <dpnp.sum>`,
:func:`swapaxes <dpnp.swapaxes>`, :func:`take <dpnp.take>`, :func:`trace <dpnp.trace>`,
:func:`transpose <dpnp.transpose>`, :func:`var <dpnp.var>`.


Array conversion
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.item
   dpnp.ndarray.tolist
   dpnp.ndarray.tobytes
   dpnp.ndarray.tofile
   dpnp.ndarray.dump
   dpnp.ndarray.dumps
   dpnp.ndarray.astype
   dpnp.ndarray.byteswap
   dpnp.ndarray.copy
   dpnp.ndarray.view
   dpnp.ndarray.getfield
   dpnp.ndarray.setflags
   dpnp.ndarray.fill
   dpnp.ndarray.get_array


Shape manipulation
------------------

For reshape, resize, and transpose, the single tuple argument may be
replaced with ``n`` integers which will be interpreted as an n-tuple.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.reshape
   dpnp.ndarray.resize
   dpnp.ndarray.transpose
   dpnp.ndarray.swapaxes
   dpnp.ndarray.flatten
   dpnp.ndarray.ravel
   dpnp.ndarray.squeeze


Item selection and manipulation
-------------------------------

For array methods that take an *axis* keyword, it defaults to
*None*. If axis is *None*, then the array is treated as a 1-D
array. Any other value for *axis* represents the dimension along which
the operation should proceed.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.take
   dpnp.ndarray.put
   dpnp.ndarray.repeat
   dpnp.ndarray.choose
   dpnp.ndarray.sort
   dpnp.ndarray.argsort
   dpnp.ndarray.partition
   dpnp.ndarray.argpartition
   dpnp.ndarray.searchsorted
   dpnp.ndarray.nonzero
   dpnp.ndarray.compress
   dpnp.ndarray.diagonal


Calculation
-----------

Many of these methods take an argument named *axis*. In such cases,

- If *axis* is *None* (the default), the array is treated as a 1-D array and
  the operation is performed over the entire array. This behavior is also the
  default if *self* is a 0-dimensional array.

- If *axis* is an integer, then the operation is done over the given axis (for
  each 1-D subarray that can be created along the given axis).

The parameter *dtype* specifies the data type over which a reduction operation
(like summing) should take place. The default reduce data type is the same as
the data type of *self*. To avoid overflow, it can be useful to perform the
reduction using a larger data type.

For several methods, an optional *out* argument can also be provided and the
result will be placed into the output array given. The *out* argument must be
an :class:`dpnp.ndarray` and have the same number of elements as the result
array. It can have a different data type in which case casting will be
performed.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.max
   dpnp.ndarray.argmax
   dpnp.ndarray.min
   dpnp.ndarray.argmin
   dpnp.ndarray.clip
   dpnp.ndarray.conj
   dpnp.ndarray.conjugate
   dpnp.ndarray.round
   dpnp.ndarray.trace
   dpnp.ndarray.sum
   dpnp.ndarray.cumsum
   dpnp.ndarray.mean
   dpnp.ndarray.var
   dpnp.ndarray.std
   dpnp.ndarray.prod
   dpnp.ndarray.cumprod
   dpnp.ndarray.all
   dpnp.ndarray.any


Arithmetic, matrix multiplication, and comparison operations
------------------------------------------------------------

Arithmetic and comparison operations on :class:`dpnp.ndarrays <dpnp.ndarray>`
are defined as element-wise operations, and generally yield
:class:`dpnp.ndarray` objects as results.

Each of the arithmetic operations (``+``, ``-``, ``*``, ``/``, ``//``, ``%``,
``divmod()``, ``**`` or ``pow()``, ``<<``, ``>>``, ``&``, ``^``, ``|``, ``~``)
and the comparisons (``==``, ``<``, ``>``, ``<=``, ``>=``, ``!=``) is
equivalent to the corresponding universal function (or :term:`ufunc` for short)
in DPNP. For more information, see the section on :ref:`Universal Functions
<ufuncs>`.


Comparison operators:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__lt__
   dpnp.ndarray.__le__
   dpnp.ndarray.__gt__
   dpnp.ndarray.__ge__
   dpnp.ndarray.__eq__
   dpnp.ndarray.__ne__

Truth value of an array (:class:`bool() <bool>`):

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__bool__

.. note::

   Truth-value testing of an array invokes
   :meth:`dpnp.ndarray.__bool__`, which raises an error if the number of
   elements in the array is not 1, because the truth value
   of such arrays is ambiguous. Use :meth:`.any() <dpnp.ndarray.any>` and
   :meth:`.all() <dpnp.ndarray.all>` instead to be clear about what is meant
   in such cases. (If you wish to check for whether an array is empty,
   use for example ``.size > 0``.)


Unary operations:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__neg__
   dpnp.ndarray.__pos__
   dpnp.ndarray.__abs__
   dpnp.ndarray.__invert__


Arithmetic:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__add__
   dpnp.ndarray.__sub__
   dpnp.ndarray.__mul__
   dpnp.ndarray.__truediv__
   dpnp.ndarray.__floordiv__
   dpnp.ndarray.__mod__
   dpnp.ndarray.__divmod__
   dpnp.ndarray.__pow__
   dpnp.ndarray.__lshift__
   dpnp.ndarray.__rshift__
   dpnp.ndarray.__and__
   dpnp.ndarray.__or__
   dpnp.ndarray.__xor__


Arithmetic, reflected:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__radd__
   dpnp.ndarray.__rsub__
   dpnp.ndarray.__rmul__
   dpnp.ndarray.__rtruediv__
   dpnp.ndarray.__rfloordiv__
   dpnp.ndarray.__rmod__
   dpnp.ndarray.__rpow__
   dpnp.ndarray.__rlshift__
   dpnp.ndarray.__rrshift__
   dpnp.ndarray.__rand__
   dpnp.ndarray.__ror__
   dpnp.ndarray.__rxor__


Arithmetic, in-place:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__iadd__
   dpnp.ndarray.__isub__
   dpnp.ndarray.__imul__
   dpnp.ndarray.__itruediv__
   dpnp.ndarray.__ifloordiv__
   dpnp.ndarray.__imod__
   dpnp.ndarray.__ipow__
   dpnp.ndarray.__ilshift__
   dpnp.ndarray.__irshift__
   dpnp.ndarray.__iand__
   dpnp.ndarray.__ior__
   dpnp.ndarray.__ixor__


Matrix Multiplication:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__matmul__
   dpnp.ndarray.__rmatmul__
   dpnp.ndarray.__imatmul__


Special methods
---------------

For standard library functions:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__copy__
   dpnp.ndarray.__deepcopy__
   .. dpnp.ndarray.__reduce__
   dpnp.ndarray.__setstate__

Basic customization:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__new__
   dpnp.ndarray.__array__
   dpnp.ndarray.__array_namespace__
   dpnp.ndarray.__array_wrap__
   dpnp.ndarray.__dlpack__
   dpnp.ndarray.__dlpack_device__

Container customization: (see :ref:`Indexing <routines.indexing>`)

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__len__
   dpnp.ndarray.__iter__
   dpnp.ndarray.__getitem__
   dpnp.ndarray.__setitem__
   dpnp.ndarray.__contains__

Conversion; the operations :class:`int() <int>`, :class:`float() <float>`,
:class:`complex() <complex>` and :func:`operator.index() <operator.index>`.
They work only on arrays that have one element in them
and return the appropriate scalar.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__index__
   dpnp.ndarray.__int__
   dpnp.ndarray.__float__
   dpnp.ndarray.__complex__

String representations:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__str__
   dpnp.ndarray.__repr__
