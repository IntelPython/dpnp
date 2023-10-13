.. _dptcl:
.. include:: ./ext_links.txt

Interplay with the Data Parallel Control Library
===============================================

`Data Parallel Control Library`_ provides API to manage specific
`SYCL*`_ resources for SYCL-based Python packages.

An example below demonstrates how the Data Parallel Extension for NumPy* can be
easily combined with the device management interface provided by dpctl package.

Literally, the SYCL* queue manager interface from the dpctl package allows
to set an input queue as the currently usable queue inside the context
manager's scope. This way an array creation function from the dpnp package
which is defined inside the context will allocate the data using that queue.

.. code-block:: python
  :linenos:

  import dpctl
  import dpnp as np

  with dpctl.device_context("opencl:gpu"):
      x = np.array([1, 2, 3])
      s = np.sum(x)

For more information please refer to `Data Parallel Control Library`_
documentation.

Example
-------
.. literalinclude:: ../examples/example10.py
  :linenos:
  :language: python
  :lines: 35-
