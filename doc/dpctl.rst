.. _dptcl:
.. include:: ./ext_links.txt

Interplay with the Data Parallel Control Library
===============================================

`Data Parallel Control Library`_ provides API to manage specific
`SYCL*`_ resources for SYCL-based Python packages.

An example below demonstrates how the Data Parallel Extension for NumPy* can be
easily combined with the device management interface provided by dpctl package.

.. code-block:: python
  :linenos:

      import dpctl
      import dpnp

      d = dpctl.select_cpu_device()
      x = dpnp.array([1, 2, 3], device=d)
      s = dpnp.sum(x)

      y = dpnp.linspace(0, dpnp.pi, num=10**6, device="gpu")
      f = 1 + y * dpnp.sin(y)

      # locate argument where function attains global maximum
      max_arg = x[dpnp.argmax(f)]
      max_val = dpnp.max(f)


For more information please refer to `Data Parallel Control Library`_
documentation.

Example
-------
.. literalinclude:: ../examples/example10.py
  :linenos:
  :language: python
  :lines: 35-
