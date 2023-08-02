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
:ref:`Array Creation Routines <routines.creation>`, and also by using the low-level
:class:`dpnp.ndarray` constructor:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray


Indexing arrays
---------------

Arrays can be indexed using an extended Python slicing syntax,
``array[selection]``.  Similar syntax is also used for accessing
fields in a :term:`structured data type`.

.. seealso:: :ref:`Array Indexing Routines <routines.indexing>`.


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
   dpnp.ndarray.base


Data type
---------

.. seealso:: :ref:`Data type objects <dtype>`

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
   dpnp.ndarray.real
   dpnp.ndarray.imag
   dpnp.ndarray.flat


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
:func:`prod <dpnp.prod>`, :func:`ptp <dpnp.ptp>`, :func:`put <dpnp.put>`,
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
   dpnp.ndarray.itemset
   dpnp.ndarray.tostring
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

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.max
   dpnp.ndarray.argmax
   dpnp.ndarray.min
   dpnp.ndarray.argmin
   dpnp.ndarray.ptp
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

Each of the arithmetic operations (``+``, ``-``, ``*``, ``/``, ``//``,
``%``, ``divmod()``, ``**`` or ``pow()``, ``<<``, ``>>``, ``&``,
``^``, ``|``, ``~``) and the comparisons (``==``, ``<``, ``>``,
``<=``, ``>=``, ``!=``) is equivalent to the corresponding
universal function (or **ufunc** for short) in DPNP.  For
more information, see the section on :ref:`Universal Functions
<ufunc>`.


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


Special methods
---------------

For standard library functions:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__copy__
   dpnp.ndarray.__deepcopy__
   dpnp.ndarray.__reduce__
   dpnp.ndarray.__setstate__

Basic customization:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__new__
   dpnp.ndarray.__array__
   dpnp.ndarray.__array_wrap__

Container customization: (see :ref:`Indexing <routines.indexing>`)

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__len__
   dpnp.ndarray.__getitem__
   dpnp.ndarray.__setitem__
   dpnp.ndarray.__contains__

Conversion; the operations :class:`int() <int>`,
:class:`float() <float>` and :class:`complex() <complex>`.
They work only on arrays that have one element in them
and return the appropriate scalar.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__int__
   dpnp.ndarray.__float__
   dpnp.ndarray.__complex__

String representations:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dpnp.ndarray.__str__
   dpnp.ndarray.__repr__
