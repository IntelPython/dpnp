.. currentmodule:: dpnp

The N-dimensional array (:class:`ndarray <ndarray>`)
=========================================================

:class:`ndarray` is the DPNP counterpart of NumPy :class:`numpy.ndarray`.

For the basic concept of ``ndarray``\s, please refer to the `NumPy documentation <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray
   dpnp_array.dpnp_array


Constructing arrays
-------------------

New arrays can be constructed using the routines detailed in
:ref:`Array Creation Routines <routines.array-creation>`, and also by using the low-level
:class:`ndarray` constructor:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray


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

   ndarray.flags
   ndarray.shape
   ndarray.strides
   ndarray.ndim
   ndarray.data
   ndarray.size
   ndarray.itemsize
   ndarray.nbytes
   ndarray.device
   ndarray.sycl_context
   ndarray.sycl_device
   ndarray.sycl_queue
   ndarray.usm_type


Data type
---------

.. seealso:: :ref:`Available array data types <Data types>`

The data type object associated with the array can be found in the
:attr:`dtype <ndarray.dtype>` attribute:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.dtype


Other attributes
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.T
   ndarray.mT
   ndarray.real
   ndarray.imag
   ndarray.flat


Special attributes
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__sycl_usm_array_interface__
   ndarray.__usm_ndarray__


Array methods
-------------

An :class:`ndarray` object has many methods which operate on or with
the array in some fashion, typically returning an array result. These
methods are briefly explained below. (Each method's docstring has a
more complete description.)

For the following methods there are also corresponding functions in
:mod:`dpnp`: :func:`all <all>`, :func:`any <any>`,
:func:`argmax <argmax>`, :func:`argmin <argmin>`,
:func:`argpartition <argpartition>`, :func:`argsort <argsort>`,
:func:`choose <choose>`, :func:`clip <clip>`,
:func:`compress <compress>`, :func:`copy <copy>`,
:func:`cumprod <cumprod>`, :func:`cumsum <cumsum>`,
:func:`diagonal <diagonal>`, :func:`imag <imag>`,
:func:`max <max>`, :func:`mean <mean>`, :func:`min <min>`,
:func:`nonzero <nonzero>`, :func:`partition <partition>`,
:func:`prod <prod>`, :func:`put <put>`,
:func:`ravel <ravel>`, :func:`real <real>`, :func:`repeat <repeat>`,
:func:`reshape <reshape>`, :func:`round <around>`,
:func:`searchsorted <searchsorted>`, :func:`sort <sort>`,
:func:`squeeze <squeeze>`, :func:`std <std>`, :func:`sum <sum>`,
:func:`swapaxes <swapaxes>`, :func:`take <take>`, :func:`trace <trace>`,
:func:`transpose <transpose>`, :func:`var <var>`.


Array conversion
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.item
   ndarray.tolist
   ndarray.tobytes
   ndarray.tofile
   ndarray.dump
   ndarray.dumps
   ndarray.astype
   ndarray.byteswap
   ndarray.copy
   ndarray.view
   ndarray.getfield
   ndarray.setflags
   ndarray.fill
   ndarray.get_array


Shape manipulation
------------------

For reshape, resize, and transpose, the single tuple argument may be
replaced with ``n`` integers which will be interpreted as an n-tuple.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.reshape
   ndarray.resize
   ndarray.transpose
   ndarray.swapaxes
   ndarray.flatten
   ndarray.ravel
   ndarray.squeeze


Item selection and manipulation
-------------------------------

For array methods that take an *axis* keyword, it defaults to
*None*. If axis is *None*, then the array is treated as a 1-D
array. Any other value for *axis* represents the dimension along which
the operation should proceed.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.take
   ndarray.put
   ndarray.repeat
   ndarray.choose
   ndarray.sort
   ndarray.argsort
   ndarray.partition
   ndarray.argpartition
   ndarray.searchsorted
   ndarray.nonzero
   ndarray.compress
   ndarray.diagonal


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
an :class:`ndarray` and have the same number of elements as the result
array. It can have a different data type in which case casting will be
performed.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.max
   ndarray.argmax
   ndarray.min
   ndarray.argmin
   ndarray.clip
   ndarray.conj
   ndarray.conjugate
   ndarray.round
   ndarray.trace
   ndarray.sum
   ndarray.cumsum
   ndarray.mean
   ndarray.var
   ndarray.std
   ndarray.prod
   ndarray.cumprod
   ndarray.all
   ndarray.any


Arithmetic, matrix multiplication, and comparison operations
------------------------------------------------------------

Arithmetic and comparison operations on :class:`ndarrays <ndarray>`
are defined as element-wise operations, and generally yield
:class:`ndarray` objects as results.

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

   ndarray.__lt__
   ndarray.__le__
   ndarray.__gt__
   ndarray.__ge__
   ndarray.__eq__
   ndarray.__ne__

Truth value of an array (:class:`bool() <bool>`):

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__bool__

.. note::

   Truth-value testing of an array invokes
   :meth:`ndarray.__bool__`, which raises an error if the number of
   elements in the array is not 1, because the truth value
   of such arrays is ambiguous. Use :meth:`.any() <ndarray.any>` and
   :meth:`.all() <ndarray.all>` instead to be clear about what is meant
   in such cases. (If you wish to check for whether an array is empty,
   use for example ``.size > 0``.)


Unary operations:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__neg__
   ndarray.__pos__
   ndarray.__abs__
   ndarray.__invert__


Arithmetic:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__add__
   ndarray.__sub__
   ndarray.__mul__
   ndarray.__truediv__
   ndarray.__floordiv__
   ndarray.__mod__
   ndarray.__divmod__
   ndarray.__pow__
   ndarray.__lshift__
   ndarray.__rshift__
   ndarray.__and__
   ndarray.__or__
   ndarray.__xor__


Arithmetic, reflected:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__radd__
   ndarray.__rsub__
   ndarray.__rmul__
   ndarray.__rtruediv__
   ndarray.__rfloordiv__
   ndarray.__rmod__
   ndarray.__rpow__
   ndarray.__rlshift__
   ndarray.__rrshift__
   ndarray.__rand__
   ndarray.__ror__
   ndarray.__rxor__


Arithmetic, in-place:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__iadd__
   ndarray.__isub__
   ndarray.__imul__
   ndarray.__itruediv__
   ndarray.__ifloordiv__
   ndarray.__imod__
   ndarray.__ipow__
   ndarray.__ilshift__
   ndarray.__irshift__
   ndarray.__iand__
   ndarray.__ior__
   ndarray.__ixor__


Matrix Multiplication:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__matmul__
   ndarray.__rmatmul__
   ndarray.__imatmul__


Special methods
---------------

For standard library functions:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__copy__
   ndarray.__deepcopy__
   .. ndarray.__reduce__
   ndarray.__setstate__

Basic customization:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__new__
   ndarray.__array__
   ndarray.__array_namespace__
   ndarray.__array_wrap__
   ndarray.__dlpack__
   ndarray.__dlpack_device__

Container customization: (see :ref:`Indexing <routines.indexing>`)

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__len__
   ndarray.__iter__
   ndarray.__getitem__
   ndarray.__setitem__
   ndarray.__contains__

Conversion; the operations :class:`int() <int>`, :class:`float() <float>`,
:class:`complex() <complex>` and :func:`operator.index() <operator.index>`.
They work only on arrays that have one element in them
and return the appropriate scalar.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__bytes__
   ndarray.__index__
   ndarray.__int__
   ndarray.__float__
   ndarray.__complex__

String representations:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ndarray.__str__
   ndarray.__repr__
   ndarray.__format__
