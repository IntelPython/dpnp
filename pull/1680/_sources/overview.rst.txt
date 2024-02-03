.. _overview:
.. include:: ./ext_links.txt

Overview
========

.. module:: dpnp

The Data Parallel Extension for NumPy* (dpnp package) - a library that
implements a subset of `NumPy*`_ that can be executed on any
data parallel device. The subset is a drop-in replacement of core `NumPy*`_
functions and numerical data types.

The Data Parallel Extension for NumPy* is being developed as part of
`Intel AI Analytics Toolkit`_ and is distributed with the
`Intel Distribution for Python*`_. The dpnp package is also available
on Anaconda cloud. Please refer the :doc:`quick_start_guide` page to learn more.

Being drop-in replacement for `NumPy*`_ means that the usage is very similar:

   >>> import dpnp as np

The :class:`dpnp.ndarray` class is a compatible alternative of
:class:`numpy.ndarray`.

   >>> x = np.array([1, 2, 3])

``x`` in the above example is an instance of :class:`dpnp.ndarray` that
is created identically to ``NumPy*``'s one. The key difference of
:class:`dpnp.ndarray` from :class:`numpy.ndarray` is that the memory
is allocated on the default `SYCL*`_ device, which is a ``"gpu"`` on systems
with integrated or discrete GPU (otherwise it is the ``"host"`` device
on systems that do not have GPU).

Most of the array manipulations are also done in the way similar to `NumPy*`_ such as:

   >>> s = np.sum(x)

Please see the :ref:`API Reference <dpnp_reference>` for the complete list of supported `NumPy*`_ APIs
along with their limitations.
