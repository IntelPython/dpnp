.. _routines.exceptions:

Exceptions and Warnings
=======================

General exceptions used by DPNP. Note that some exceptions may be module
specific, such as linear algebra errors.

.. currentmodule:: dpnp.exceptions

Exceptions
----------

.. data:: AxisError

    Given when an axis is invalid.

.. data:: DLPackCreationError

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
