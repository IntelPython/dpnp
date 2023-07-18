DPCtl Usage
===========

`DPCtl <https://github.com/IntelPython/dpctl>`_ provides API to manage
specific SYCL resources for SYCL-based Python packages. DPNP uses DPCtl as
a global SYCL queue manager. Below code illustrates simple usage of DPNP
in combination with dpCtl.

.. code-block:: python
  :linenos:

  import dpctl
  import dpnp as np

  with dpctl.device_context("opencl:gpu"):
      x = np.array([1, 2, 3])
      s = np.sum(x)

For more information please refer to `DPCtl's documentation <https://intelpython.github.io/dpctl>`_.

Example
~~~~~~~
.. literalinclude:: ../examples/example10.py
  :linenos:
  :language: python
  :lines: 35-
