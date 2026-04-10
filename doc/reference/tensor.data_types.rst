.. _dpnp_tensor_data_types:

.. currentmodule:: dpnp.tensor

Data types
==========

:py:mod:`dpnp.tensor` supports the following data types:

+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Data Type      | Description                                                                                                                                                                             |
+================+=========================================================================================================================================================================================+
| ``bool``       | Boolean (``True`` or ``False``)                                                                                                                                                         |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``int8``       | An 8-bit signed integer type capable of representing :math:`v` subject to :math:`-2^7 \le v < 2^7`                                                                                      |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``int16``      | A 16-bit signed integer type capable of representing :math:`v` subject to :math:`-2^{15} \le v < 2^{15}`                                                                                |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``int32``      | A 32-bit signed integer type capable of representing :math:`v` subject to :math:`-2^{31} \le v < 2^{31}`                                                                                |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``int64``      | A 64-bit signed integer type capable of representing :math:`v` subject to :math:`-2^{63} \le v < 2^{63}`                                                                                |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``uint8``      | An 8-bit unsigned integer type capable of representing :math:`v` subject to :math:`0 \le v < 2^8`                                                                                       |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``uint16``     | A 16-bit unsigned integer type capable of representing :math:`v` subject to :math:`0 \le v < 2^{16}`                                                                                    |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``uint32``     | A 32-bit unsigned integer type capable of representing :math:`v` subject to :math:`0 \le v < 2^{32}`                                                                                    |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``uint64``     | A 64-bit unsigned integer type capable of representing :math:`v` subject to :math:`0 \le v < 2^{64}`                                                                                    |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``float16``    | An IEEE-754 half-precision (16-bit) binary floating-point number (see `IEEE 754-2019`_)                                                                                                 |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``float32``    | An IEEE-754 single-precision (32-bit) binary floating-point number (see `IEEE 754-2019`_)                                                                                               |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``float64``    | An IEEE-754 double-precision (64-bit) binary floating-point number (see `IEEE 754-2019`_)                                                                                               |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``complex64``  | Single-precision (64-bit) complex floating-point number whose real and imaginary components are IEEE 754 single-precision (32-bit) binary floating-point numbers (see `IEEE 754-2019`_) |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``complex128`` | Double-precision (128-bit) complex floating-point number whose real and imaginary components are IEEE 754 double-precision (64-bit) binary floating-point numbers (see `IEEE 754-2019`_)|
+----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. _IEEE 754-2019: https://doi.org/10.1109%2FIEEESTD.2019.8766229

Data type support by array object :py:class:`usm_ndarray` depends on capabilities of :class:`dpctl.SyclDevice` where array is allocated.

Half-precision floating-point type ``float16`` is supported only for devices whose attribute :attr:`dpctl.SyclDevice.has_aspect_fp16` evaluates to ``True``.

Double-precision floating-point type ``float64`` and double-precision complex floating-point type ``complex128`` are supported only for devices whose attribute :attr:`dpctl.SyclDevice.has_aspect_fp64`
evaluates to ``True``.

If prerequisites are not met, requests to create an instance of an array object for these types will raise an exception.

Data type objects are instances of :py:class:`dtype` object, and support equality comparison by implementing
special method :meth:`__eq__`.

.. py:class:: dtype

    Same as :py:class:`numpy.dtype`

    .. py:method:: __eq__

        Check if data-type instances are equal.


Default integral data type
--------------------------

The default integral data type is :attr:`int64` for all supported devices.

Default indexing data type
--------------------------

The default indexing data type is :attr:`int64` for all supported devices.

Default real floating-point data type
-------------------------------------

The default real floating-point type depends on the capabilities of device where array is allocated.
If the device support double precision floating-point types, the default real floating-point type
is :attr:`float64`, otherwise :attr:`float32`.

Make sure to select an appropriately capable device for an application that requires use of double
precision floating-point type.

Default complex floating-point data type
----------------------------------------

Like for the default real floating-point type, the default complex floating-point type depends on
capabilities of device. If the device support double precision real floating-point types, the default
complex floating-point type is :attr:`complex128`, otherwise :attr:`complex64`.


Querying default data types programmatically
--------------------------------------------

The data type can be discovered programmatically using Array API :ref:`inspection functions <dpnp_tensor_inspection>`:

.. code-block:: python

    import dpctl
    from dpnp import tensor

    device = dpctl.select_default_device()
    # get default data types for default-selected device
    default_types = tensor.__array_namespace_info__().default_dtypes(device)
    int_dt = default_types["integral"]
    ind_dt = default_types["indexing"]
    rfp_dt = default_types["real floating"]
    cfp_dt = default_types["complex floating"]


Type promotion rules
--------------------

Type promotion rules govern the behavior of an array library when a function does not have
a dedicated implementation for the data type(s) of the input array(s).

In such a case, input arrays may be cast to data types for which a dedicated implementation
exists. For example, when :data:`sin` is applied to array of integral values.

Type promotion rules used in :py:mod:`dpnp.tensor` are consistent with the
Python Array API specification's `type promotion rules <https://data-apis.org/array-api/latest/API_specification/type_promotion.html>`_
for devices that support double precision floating-point type.


For devices that do not support double precision floating-point type, the type promotion rule is
truncated by removing nodes corresponding to unsupported data types and edges that lead to them.
