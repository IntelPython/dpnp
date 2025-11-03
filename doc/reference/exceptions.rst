.. _routines.exceptions:

.. py:module:: dpnp.exceptions

Exceptions and Warnings
=======================

.. hint:: `NumPy API Reference: Exceptions and Warnings <https://numpy.org/doc/stable/reference/routines.exceptions.html>`_

General exceptions used by DPNP. Note that some exceptions may be module
specific, such as linear algebra errors.

Exceptions
----------

.. exception:: AxisError

   Given when an axis is invalid.

.. exception:: DLPackCreationError

   Given when constructing DLPack capsule from either :class:`dpnp.ndarray` or
   :class:`dpctl.tensor.usm_ndarray` based on a USM allocation
   on a partitioned SYCL device.

   .. rubric:: Examples

   .. code-block:: python

      >>> import dpnp as np
      >>> import dpctl
      >>> dev = dpctl.SyclDevice('cpu')
      >>> sdevs = dev.create_sub_devices(partition=[1, 1])
      >>> q = dpctl.SyclQueue(sdevs[0])
      >>> x = np.ones(10, sycl_queue=q)
      >>> np.from_dlpack(x)
      Traceback (most recent call last):
      ...
      DLPackCreationError: to_dlpack_capsule: DLPack can only export arrays based on USM allocations bound to a default platform SYCL context

.. exception:: ExecutionPlacementError

   Given when execution placement target can not be unambiguously determined
   from input arrays. Make sure that input arrays are associated with the same
   :class:`dpctl.SyclQueue`, or migrate data to the same
   :class:`dpctl.SyclQueue` using :meth:`dpnp.ndarray.to_device` method.

.. exception:: SyclContextCreationError

   Given when :class:`dpctl.SyclContext` instance could not be created.

.. exception:: SyclDeviceCreationError

   Given when :class:`dpctl.SyclDevice` instance could not be created.

.. exception:: SyclQueueCreationError

   Given when :class:`dpctl.SyclQueue` instance could not be created.
   The creation can fail if the filter string is invalid, or the backend or
   device type values are not supported.

.. exception:: USMAllocationError

   Given when Unified Shared Memory (USM) allocation call returns a null
   pointer, signaling a failure to perform the allocation.
   Some common reasons for allocation failure are:

      * insufficient free memory to perform the allocation request
      * allocation size exceeds the maximum supported by targeted backend


.. automodule:: dpnp.exceptions
    :no-index:
